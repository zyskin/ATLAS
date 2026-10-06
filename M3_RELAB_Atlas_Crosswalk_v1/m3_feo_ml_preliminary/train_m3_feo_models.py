#!/usr/bin/env python3
"""Evaluate preliminary FeO regressors with Apollo-parent-grouped validation.

The split unit is the Apollo parent sample, not the individual spectrum.  This
prevents spectra of the same parent/sample family from appearing in both the
training and test portions of a fold.  The script reports honest out-of-parent
predictions for three published/legacy baselines and four small-data models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


VERSION = "m3-feo-regression-1.0"
BAND_COLUMNS = [f"m3_b{band:02d}" for band in range(1, 75)]
BASELINES = {
    "lucey_direct_b09_b19": "lucey_direct_feo_wt_percent",
    "kumar_m3_mean_linear": "kumar_m3_linear_feo_wt_percent",
    "kumar_m3_mean_power": "kumar_m3_power_feo_wt_percent",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def flatten_prediction(value: object) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(-1)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    actual = y_true[valid]
    predicted = y_pred[valid]
    if len(actual) == 0:
        return {
            "n": 0, "mae": math.nan, "rmse": math.nan,
            "median_absolute_error": math.nan, "bias": math.nan, "r2": math.nan,
        }
    residual = predicted - actual
    return {
        "n": int(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(actual, predicted))),
        "median_absolute_error": float(np.median(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "r2": float(r2_score(actual, predicted)) if len(actual) >= 2 else math.nan,
    }


def model_definitions(search_level: str, seed: int, max_pls: int):
    ridge_grid = {
        "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]
        if search_level == "standard" else [0.1, 1.0, 10.0]
    }
    pls_values = [value for value in ([2, 4, 6, 8, 12, 16] if search_level == "standard"
                                      else [2, 4, 8]) if value <= max_pls]
    if not pls_values:
        pls_values = [1]
    svr_grid = ({
        "model__C": [1.0, 10.0, 100.0],
        "model__epsilon": [0.05, 0.2, 0.5],
        "model__gamma": ["scale", 0.001, 0.01],
    } if search_level == "standard" else {
        "model__C": [1.0, 10.0],
        "model__epsilon": [0.1, 0.5],
        "model__gamma": ["scale"],
    })
    trees_grid = ({
        "model__n_estimators": [400],
        "model__max_features": [0.35, 0.7, 1.0],
        "model__min_samples_leaf": [1, 2, 4],
    } if search_level == "standard" else {
        "model__n_estimators": [300],
        "model__max_features": [0.5, 1.0],
        "model__min_samples_leaf": [1, 3],
    })
    return {
        "ridge": (
            Pipeline([("scale", StandardScaler()), ("model", Ridge())]),
            ridge_grid,
        ),
        "pls": (
            Pipeline([
                ("scale", StandardScaler()),
                ("model", PLSRegression(scale=False, max_iter=1000)),
            ]),
            {"model__n_components": pls_values},
        ),
        "svr_rbf": (
            Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="rbf"))]),
            svr_grid,
        ),
        "extra_trees": (
            Pipeline([("model", ExtraTreesRegressor(random_state=seed, n_jobs=-1))]),
            trees_grid,
        ),
    }


def tune(estimator, parameter_grid, x, y, groups, requested_splits: int):
    unique_groups = np.unique(groups)
    splits = min(requested_splits, len(unique_groups))
    if splits < 2:
        # This should occur only in an extremely small outer-training set.
        first = {name: values[0] for name, values in parameter_grid.items()}
        return clone(estimator).set_params(**first).fit(x, y), first
    cross_validation = GroupKFold(n_splits=splits)
    parameter_grid = dict(parameter_grid)
    if "model__n_components" in parameter_grid:
        # A PLS choice must also fit the smallest inner-training partition, not
        # merely the complete outer-training partition.
        smallest_training_set = min(
            len(train_index)
            for train_index, _ in cross_validation.split(x, y, groups)
        )
        allowed = [value for value in parameter_grid["model__n_components"]
                   if value <= max(1, min(x.shape[1], smallest_training_set - 1))]
        parameter_grid["model__n_components"] = allowed or [1]
    search = GridSearchCV(
        clone(estimator), parameter_grid, scoring="neg_mean_absolute_error",
        cv=cross_validation, n_jobs=-1, refit=True,
        error_score="raise",
    )
    search.fit(x, y, groups=groups)
    return search.best_estimator_, search.best_params_


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--search-level", choices=("quick", "standard"), default="quick")
    parser.add_argument("--seed", type=int, default=20261006)
    args = parser.parse_args()

    frame = pd.read_csv(args.dataset, dtype={"i_atlas_sample": str})
    required = {"id", "i_atlas_sample", "feo_mean_wt_percent", *BAND_COLUMNS,
                *BASELINES.values()}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(sorted(missing))}")
    if frame["id"].duplicated().any():
        duplicate = frame.loc[frame["id"].duplicated(), "id"].iloc[0]
        raise ValueError(f"Dataset has duplicate stable id: {duplicate}")

    x = frame[BAND_COLUMNS].to_numpy(dtype=float)
    y = frame["feo_mean_wt_percent"].to_numpy(dtype=float)
    groups = frame["i_atlas_sample"].astype(str).to_numpy()
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("M3 features and FeO target must all be finite")
    group_count = len(np.unique(groups))
    outer_splits = min(args.outer_folds, group_count)
    if outer_splits < 3:
        raise ValueError(f"Need at least 3 Apollo-parent groups; found {group_count}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    splitter = GroupKFold(n_splits=outer_splits)
    definitions = model_definitions(
        args.search_level, args.seed,
        max_pls=max(1, min(len(frame) - 1, len(BAND_COLUMNS))),
    )

    predictions: dict[str, np.ndarray] = {}
    predictions["training_parent_mean"] = np.full(len(frame), np.nan)
    for name, column in BASELINES.items():
        predictions[name] = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    for name in definitions:
        predictions[name] = np.full(len(frame), np.nan)

    fold_number = np.zeros(len(frame), dtype=int)
    fold_parameters: dict[str, list[dict[str, object]]] = {name: [] for name in definitions}
    for fold, (train_index, test_index) in enumerate(splitter.split(x, y, groups), 1):
        fold_number[test_index] = fold
        predictions["training_parent_mean"][test_index] = float(np.mean(y[train_index]))
        train_groups = groups[train_index]
        max_components = max(1, min(len(train_index) - 1, x.shape[1]))
        fold_defs = model_definitions(args.search_level, args.seed, max_components)
        for name, (estimator, grid) in fold_defs.items():
            fitted, best = tune(
                estimator, grid, x[train_index], y[train_index], train_groups,
                args.inner_folds,
            )
            predictions[name][test_index] = flatten_prediction(fitted.predict(x[test_index]))
            fold_parameters[name].append({"fold": fold, "parameters": best})
        print(f"Completed grouped outer fold {fold}/{outer_splits}")

    metric_rows = []
    for name, values in predictions.items():
        metric_rows.append({"method": name, **metrics(y, values)})
    metrics_frame = pd.DataFrame(metric_rows).sort_values(["mae", "method"])
    metrics_frame.to_csv(args.output_dir / "cv_metrics.csv", index=False)

    identity = [
        "id", "i_atlas_sample", "j_child", "relab_measurement_id",
        "relab_specimen_id", "feo_mean_wt_percent", "feo_claim_count",
        "feo_values_wt_percent",
    ]
    identity = [column for column in identity if column in frame.columns]
    prediction_frame = frame[identity].copy()
    prediction_frame.insert(len(identity), "grouped_cv_fold", fold_number)
    for name, values in predictions.items():
        prediction_frame[f"prediction_{name}"] = values
        prediction_frame[f"residual_{name}"] = values - y
    prediction_frame.to_csv(args.output_dir / "cv_predictions.csv", index=False)

    # Tune and retain deployable fits on all available rows.  These are not the
    # source of the reported CV scores; their purpose is later prediction.
    final_models = {}
    final_parameters = {}
    importance_rows = []
    for name, (estimator, grid) in definitions.items():
        fitted, best = tune(estimator, grid, x, y, groups, args.inner_folds)
        final_models[name] = fitted
        final_parameters[name] = best
        joblib.dump(fitted, args.output_dir / f"model_{name}.joblib")
        model = fitted.named_steps["model"]
        if name == "ridge":
            for band, value in zip(BAND_COLUMNS, flatten_prediction(model.coef_)):
                importance_rows.append({"model": name, "feature": band, "value": value,
                                        "meaning": "standardized coefficient"})
        elif name == "extra_trees":
            for band, value in zip(BAND_COLUMNS, model.feature_importances_):
                importance_rows.append({"model": name, "feature": band, "value": value,
                                        "meaning": "impurity importance"})
    pd.DataFrame(importance_rows).to_csv(args.output_dir / "feature_importance.csv", index=False)
    (args.output_dir / "best_parameters.json").write_text(
        json.dumps({"final": final_parameters, "outer_folds": fold_parameters}, indent=2)
        + "\n", encoding="utf-8",
    )

    methods = list(metrics_frame["method"])
    columns = 3
    rows = math.ceil(len(methods) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4.8 * columns, 4.4 * rows),
                                squeeze=False)
    finite_arrays = [y[np.isfinite(y)]]
    finite_arrays.extend(values[np.isfinite(values)] for values in predictions.values())
    finite_values = np.concatenate([values for values in finite_arrays if len(values)])
    lower = float(np.min(finite_values))
    upper = float(np.max(finite_values))
    padding = max(0.5, 0.04 * (upper - lower))
    limits = (lower - padding, upper + padding)
    lookup = metrics_frame.set_index("method")
    for axis, name in zip(axes.flat, methods):
        values = predictions[name]
        valid = np.isfinite(values)
        axis.scatter(y[valid], values[valid], s=22, alpha=0.7, edgecolors="none")
        axis.plot(limits, limits, color="black", linewidth=1, linestyle="--")
        axis.set(xlim=limits, ylim=limits, xlabel="Mean accepted FeO (wt.%)",
                 ylabel="Predicted FeO (wt.%)", title=name)
        axis.text(0.03, 0.97,
                  f"MAE={lookup.loc[name, 'mae']:.3g}\nRMSE={lookup.loc[name, 'rmse']:.3g}\n"
                  f"R²={lookup.loc[name, 'r2']:.3g}",
                  transform=axis.transAxes, va="top")
    for axis in axes.flat[len(methods):]:
        axis.set_visible(False)
    figure.suptitle("Apollo-parent-grouped cross-validation", fontsize=14)
    figure.tight_layout()
    figure.savefig(args.output_dir / "predicted_vs_measured.png", dpi=180)
    plt.close(figure)

    manifest = {
        "schema": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset),
        "rows": int(len(frame)),
        "apollo_parent_groups": group_count,
        "features": BAND_COLUMNS,
        "target": "feo_mean_wt_percent",
        "group_column": "i_atlas_sample",
        "outer_folds": outer_splits,
        "inner_folds_requested": args.inner_folds,
        "search_level": args.search_level,
        "seed": args.seed,
        "reported_scores": "out-of-group predictions only; deployable final models are separate",
        "important_limitations": [
            "This is a preliminary small, non-independent laboratory dataset.",
            "Several spectra can share the same provisional parent-level FeO target.",
            "RELAB geometry, grain size, maturity, and sampling differ between records.",
            "Do not interpret an individual-band tree importance as mineral physics.",
        ],
    }
    (args.output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("\nGrouped cross-validation metrics:")
    print(metrics_frame.to_string(index=False))
    print(f"\nResults: {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
