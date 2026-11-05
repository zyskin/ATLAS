"""Configurable grouped cross-validation trainer for Apollo/RELAB FeO & Is/FeO.

One call to :func:`run_experiment` executes a single, fully specified
experiment: a (target, source-quality subset, feature family, model family,
target transform, seed, outer/inner fold count) configuration.  It performs

  * nested parent-grouped CV (outer folds for honest evaluation, inner folds for
    hyper-parameter tuning by parent-mean MAE),
  * all preprocessing / tuning strictly inside the training parents of each fold,
  * optional target transform whose inverse is applied before scoring,
  * out-of-fold prediction capture with row & parent IDs, fold, quality class,
  * measurement-level and Apollo-parent-mean metrics,
  * leak-free block-permutation importance (permute a block of adjacent bands /
    a feature group in the *held-out* parents of each fold),
  * a final model refit on all usable rows, saved as joblib with a JSON manifest.

Grouping key: ``i_atlas_sample`` (Apollo parent).  One parent never appears in
both train and test of any split.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, ParameterGrid

from .features import build_features, feature_block_indices
from .models import (
    DEFAULT_FEATURE_FAMILY,
    LINEAR_MODELS,
    SpectralAngleRegressor,
    build_model,
)

GROUP_COLUMN = "i_atlas_sample"
ROW_ID_COLUMN = "id"
SCOPE_COLUMN = "feo_target_scope"
EXACT_SCOPES = {"exact_relab_association_only", "mixed_exact_and_parent"}


# --------------------------------------------------------------------------- #
# Configuration + helpers                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentConfig:
    dataset: str
    target_column: str
    model: str
    feature_family: str | None = None          # defaults per-model
    subset: str = "all"                          # "all" | "exact_only"
    target_transform: str = "none"              # "none" | "log1p"
    outer_folds: int = 5
    inner_folds: int = 3
    seed: int = 20261105
    permutation_repeats: int = 5
    output_dir: str = "data/ml_experiments_v4/run"
    label: str | None = None
    save_final_model: bool = True               # False keeps validation dirs small
    randomized_folds: bool = False              # shuffle parents into folds
    fold_seed: int | None = None                # seed for randomized fold partition

    def resolved_feature_family(self) -> str:
        if self.feature_family:
            return self.feature_family
        if self.model == "spectral_angle":
            return "raw"
        return DEFAULT_FEATURE_FAMILY.get(self.model, "raw")

    def resolved_label(self) -> str:
        return self.label or f"{self.model}__{self.resolved_feature_family()}"


class RandomizedGroupKFold:
    """Grouped K-fold that shuffles whole groups into folds with a fixed seed.

    Unlike sklearn's ``GroupKFold`` (deterministic, seed-independent partition),
    this assigns each unique group to one of ``n_splits`` folds after a seeded
    shuffle, so repeated runs with different ``fold_seed`` yield genuinely
    different parent partitions while never splitting a group across train/test.
    """

    def __init__(self, n_splits: int, seed: int):
        self.n_splits = n_splits
        self.seed = seed

    def split(self, X, y=None, groups=None):
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(uniq.size)
        fold_of_group = {uniq[order[i]]: i % self.n_splits for i in range(uniq.size)}
        assign = np.array([fold_of_group[g] for g in groups])
        for k in range(self.n_splits):
            te = np.where(assign == k)[0]
            tr = np.where(assign != k)[0]
            if te.size and tr.size:
                yield tr, te


def _outer_splitter(cfg: "ExperimentConfig", n_splits: int):
    if cfg.randomized_folds:
        seed = cfg.fold_seed if cfg.fold_seed is not None else cfg.seed
        return RandomizedGroupKFold(n_splits=n_splits, seed=seed)
    return GroupKFold(n_splits=n_splits)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _target_forward(y: np.ndarray, transform: str) -> np.ndarray:
    if transform == "none":
        return y
    if transform == "log1p":
        if np.any(y < 0):
            raise ValueError("log1p transform requires non-negative targets")
        return np.log1p(y)
    raise ValueError(f"unknown transform {transform!r}")


def _target_inverse(z: np.ndarray, transform: str) -> np.ndarray:
    if transform == "none":
        return z
    if transform == "log1p":
        return np.expm1(z)
    raise ValueError(f"unknown transform {transform!r}")


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    err = predicted - actual
    abs_err = np.abs(err)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "n": int(actual.size),
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "median_absolute_error": float(np.median(abs_err)),
        "bias": float(np.mean(err)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def _parent_means(actual, predicted, groups):
    df = pd.DataFrame({"a": actual, "p": predicted, "g": groups})
    agg = df.groupby("g").mean()
    return agg["a"].to_numpy(), agg["p"].to_numpy()


# --------------------------------------------------------------------------- #
# Estimator construction (handles the custom spectral-angle model)             #
# --------------------------------------------------------------------------- #
def _make_estimator(cfg: ExperimentConfig):
    if cfg.model == "spectral_angle":
        est = SpectralAngleRegressor(degree=1, fit_origin=False)
        grid = {"degree": [1, 2], "fit_origin": [False, True]}
        return est, grid
    return build_model(cfg.model, cfg.seed)


def _fit_estimator(cfg, estimator, X, y):
    """Fit, applying target transform with inverse owned by the CV loop."""
    z = _target_forward(y, cfg.target_transform)
    estimator.fit(X, z)
    return estimator


def _predict_estimator(cfg, estimator, X):
    z = estimator.predict(X)
    return _target_inverse(np.asarray(z, dtype=np.float64), cfg.target_transform)


def _tune(cfg, X_tr, y_tr, g_tr):
    """Inner parent-grouped CV; pick params minimising parent-mean MAE."""
    base_est, grid = _make_estimator(cfg)
    combos = list(ParameterGrid(grid)) if grid else [None]
    if len(combos) <= 1:
        if combos and combos[0]:
            base_est.set_params(**combos[0])
        return base_est, (combos[0] or {})

    n_groups = np.unique(g_tr).size
    n_splits = max(2, min(cfg.inner_folds, n_groups))
    inner = GroupKFold(n_splits=n_splits)

    best_score, best_params = np.inf, combos[0]
    for params in combos:
        fold_maes = []
        ok = True
        for tr, va in inner.split(X_tr, y_tr, g_tr):
            est = clone(base_est) if hasattr(base_est, "get_params") else base_est
            try:
                est.set_params(**params)
                _fit_estimator(cfg, est, X_tr[tr], y_tr[tr])
                pred = _predict_estimator(cfg, est, X_tr[va])
            except Exception:
                ok = False
                break
            pa, pp = _parent_means(y_tr[va], pred, g_tr[va])
            fold_maes.append(np.mean(np.abs(pp - pa)))
        if ok and fold_maes:
            score = float(np.mean(fold_maes))
            if score < best_score:
                best_score, best_params = score, params
    est = clone(base_est) if hasattr(base_est, "get_params") else base_est
    est.set_params(**best_params)
    return est, best_params


# --------------------------------------------------------------------------- #
# Main experiment                                                              #
# --------------------------------------------------------------------------- #
def run_experiment(cfg: ExperimentConfig) -> dict[str, Any]:
    t0 = time.time()
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    data_path = Path(cfg.dataset)
    df = pd.read_csv(data_path)

    # Row selection: target present, optional exact-only source quality.
    mask = df[cfg.target_column].notna()
    if cfg.subset == "exact_only":
        mask &= df[SCOPE_COLUMN].isin(EXACT_SCOPES)
    sub = df.loc[mask].reset_index(drop=True)

    feature_family = cfg.resolved_feature_family()
    bands = sub[[f"m3_b{i:02d}" for i in range(1, 75)]].to_numpy(dtype=np.float64)
    X, feat_names = build_features(bands, feature_family)
    y = sub[cfg.target_column].to_numpy(dtype=np.float64)
    groups = sub[GROUP_COLUMN].to_numpy()
    row_ids = sub[ROW_ID_COLUMN].to_numpy()
    scope = sub[SCOPE_COLUMN].to_numpy() if SCOPE_COLUMN in sub else np.array(["na"] * len(sub))

    n_groups = np.unique(groups).size
    n_outer = max(2, min(cfg.outer_folds, n_groups))
    outer = _outer_splitter(cfg, n_outer)

    oof_pred = np.full(y.shape, np.nan)
    oof_fold = np.full(y.shape, -1, dtype=int)
    fold_params: list[dict] = []
    fold_rows = []
    block_defs = feature_block_indices(feat_names)
    perm_records = []

    status = "ok"
    error_msg = ""
    try:
        for k, (tr, te) in enumerate(outer.split(X, y, groups)):
            est, params = _tune(cfg, X[tr], y[tr], groups[tr])
            _fit_estimator(cfg, est, X[tr], y[tr])
            pred_te = _predict_estimator(cfg, est, X[te])
            oof_pred[te] = pred_te
            oof_fold[te] = k
            fold_params.append({"fold": k, "params": params})

            # fold metrics (measurement + parent)
            m_meas = _metrics(y[te], pred_te)
            pa, pp = _parent_means(y[te], pred_te, groups[te])
            m_par = _metrics(pa, pp)
            fold_rows.append({
                "fold": k,
                "n_train_parents": int(np.unique(groups[tr]).size),
                "n_test_parents": int(np.unique(groups[te]).size),
                "n_train_spectra": int(tr.size),
                "n_test_spectra": int(te.size),
                "mae_measurement": m_meas["mae"],
                "mae_parent": m_par["mae"],
                "rmse_parent": m_par["rmse"],
                "params": json.dumps(params),
            })

            # leak-free block permutation importance on the held-out parents.
            # Compute BOTH measurement-level and Apollo-parent-level MAE increase;
            # parent-level aggregates predictions and targets by parent *after*
            # each permutation (the metric the plots/README describe).
            g_te = groups[te]
            base_err_meas = float(np.mean(np.abs(pred_te - y[te])))
            pa0, pp0 = _parent_means(y[te], pred_te, g_te)
            base_err_parent = float(np.mean(np.abs(pp0 - pa0)))
            Xte = X[te].copy()
            for bname, cols in (block_defs.items() if cfg.permutation_repeats > 0 else []):
                incs_meas, incs_par = [], []
                for _ in range(cfg.permutation_repeats):
                    Xp = Xte.copy()
                    perm = rng.permutation(Xp.shape[0])
                    Xp[:, cols] = Xp[np.ix_(perm, cols)]
                    pred_p = _predict_estimator(cfg, est, Xp)
                    incs_meas.append(np.mean(np.abs(pred_p - y[te])) - base_err_meas)
                    pa_p, pp_p = _parent_means(y[te], pred_p, g_te)
                    incs_par.append(np.mean(np.abs(pp_p - pa_p)) - base_err_parent)
                perm_records.append({
                    "fold": k, "block": bname,
                    "mae_increase_parent_mean": float(np.mean(incs_par)),
                    "mae_increase_parent_std": float(np.std(incs_par)),
                    "mae_increase_measurement_mean": float(np.mean(incs_meas)),
                    "mae_increase_measurement_std": float(np.std(incs_meas)),
                })
    except Exception as exc:  # record and continue; caller logs failure
        status = "failed"
        error_msg = f"{type(exc).__name__}: {exc}"

    # ---- scoring on out-of-fold predictions (matched parents) ---- #
    valid = ~np.isnan(oof_pred)
    metrics_summary: dict[str, Any] = {"status": status, "error": error_msg}
    if valid.sum() > 0:
        meas = _metrics(y[valid], oof_pred[valid])
        pa, pp = _parent_means(y[valid], oof_pred[valid], groups[valid])
        par = _metrics(pa, pp)
        metrics_summary.update({
            "label": cfg.resolved_label(),
            "model": cfg.model,
            "feature_family": feature_family,
            "target_column": cfg.target_column,
            "subset": cfg.subset,
            "target_transform": cfg.target_transform,
            "n_spectra": int(valid.sum()),
            "n_parents": int(np.unique(groups[valid]).size),
            "mae_measurement": meas["mae"],
            "rmse_measurement": meas["rmse"],
            "r2_measurement": meas["r2"],
            "bias_measurement": meas["bias"],
            "mae_parent": par["mae"],
            "rmse_parent": par["rmse"],
            "median_abs_err_parent": par["median_absolute_error"],
            "bias_parent": par["bias"],
            "r2_parent": par["r2"],
        })

    # ---- persist OOF predictions ---- #
    pred_df = pd.DataFrame({
        "row_id": row_ids,
        "apollo_parent": groups,
        "scope": scope,
        "target": y,
        "oof_prediction": oof_pred,
        "fold": oof_fold,
        "model": cfg.model,
        "feature_family": feature_family,
        "target_transform": cfg.target_transform,
    })
    pred_df.to_csv(out / "oof_predictions.csv", index=False)

    if valid.sum() > 0:
        pm = pd.DataFrame({"apollo_parent": groups[valid], "target": y[valid],
                           "oof_prediction": oof_pred[valid]}).groupby("apollo_parent").mean()
        pm["abs_error"] = (pm["oof_prediction"] - pm["target"]).abs()
        pm.sort_values("abs_error", ascending=False).to_csv(out / "oof_parent_means.csv")

    pd.DataFrame(fold_rows).to_csv(out / "fold_metrics.csv", index=False)
    if perm_records:
        perm_df = pd.DataFrame(perm_records)
        perm_df.to_csv(out / "block_permutation_importance_folds.csv", index=False)
        perm_summary = (perm_df.groupby("block").agg(
            mae_increase_parent=("mae_increase_parent_mean", "mean"),
            mae_increase_parent_fold_std=("mae_increase_parent_mean", "std"),
            mae_increase_measurement=("mae_increase_measurement_mean", "mean"),
            mae_increase_measurement_fold_std=("mae_increase_measurement_mean", "std"),
        ).reset_index().sort_values("mae_increase_parent", ascending=False))
        perm_summary.to_csv(out / "block_permutation_importance.csv", index=False)

    # ---- final refit on ALL usable rows + save model ---- #
    final_params: dict = {}
    coeffs_saved = False
    if status == "ok" and valid.sum() > 0:
        final_est, final_params = _tune(cfg, X, y, groups)
        _fit_estimator(cfg, final_est, X, y)
        if cfg.save_final_model:
            joblib.dump(
                {"estimator": final_est, "feature_names": feat_names,
                 "config": dataclasses.asdict(cfg), "target_transform": cfg.target_transform},
                out / "final_model.joblib",
            )
        # standardized coefficients for linear models
        if cfg.model in LINEAR_MODELS:
            try:
                model = final_est.named_steps["model"]
                coef = np.asarray(model.coef_, dtype=np.float64).ravel()
                if coef.size == len(feat_names):
                    pd.DataFrame({"feature": feat_names, "standardized_coef": coef}) \
                        .assign(abs_coef=lambda d: d["standardized_coef"].abs()) \
                        .sort_values("abs_coef", ascending=False) \
                        .to_csv(out / "linear_standardized_coefficients.csv", index=False)
                    coeffs_saved = True
            except Exception:
                pass

    # ---- manifest ---- #
    manifest = {
        "config": dataclasses.asdict(cfg),
        "resolved_feature_family": feature_family,
        "group_column": GROUP_COLUMN,
        "n_outer_folds": int(n_outer),
        "n_inner_folds_requested": cfg.inner_folds,
        "n_feature_columns": len(feat_names),
        "fold_params": fold_params,
        "final_params": final_params,
        "metrics": metrics_summary,
        "dataset_sha256": _sha256(data_path),
        "coefficients_saved": coeffs_saved,
        "runtime_seconds": round(time.time() - t0, 2),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    import sklearn
    manifest["sklearn"] = sklearn.__version__
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    return metrics_summary
