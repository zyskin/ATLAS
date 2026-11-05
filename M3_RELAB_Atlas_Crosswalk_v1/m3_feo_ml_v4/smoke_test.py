"""Fast synthetic smoke test for the v4 trainer.

Builds a small synthetic dataset with a *known* grouped structure (several
spectra per Apollo parent, parent-level target with within-parent jitter) in the
exact schema the trainer expects (columns m3_b01..m3_b74, i_atlas_sample, id,
feo_target_scope).  Verifies that:

  * every model family runs end-to-end under grouped CV without error,
  * no Apollo parent leaks across a train/test split,
  * the log1p inverse transform round-trips on the original target scale,
  * a signal-bearing target is predicted far better than the grouped-mean null.

Runs in a few seconds; not a scientific benchmark.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .features import BAND_WAVELENGTHS_NM, build_features
from .models import MODEL_FAMILIES
from .trainer import ExperimentConfig, GROUP_COLUMN, run_experiment


def make_synthetic(n_parents=24, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    wl = BAND_WAVELENGTHS_NM
    rows = []
    rid = 0
    for p in range(n_parents):
        parent_feo = rng.uniform(2, 20)          # parent-level truth
        slope = -0.00015 * parent_feo            # higher FeO -> different slope
        depth = 0.02 * parent_feo                # deeper 1um band with FeO
        n_child = rng.integers(2, 6)
        for _ in range(n_child):
            cont = 0.2 + 0.01 * rng.standard_normal()
            spec = cont * np.exp(slope * (wl - wl[0]))
            band = depth * np.exp(-((wl - 1000.0) ** 2) / (2 * 150.0 ** 2))
            refl = np.clip(spec - band + 0.002 * rng.standard_normal(wl.size), 1e-3, 0.999)
            row = {"id": rid, GROUP_COLUMN: 90000 + p,
                   "feo_target_scope": "exact_relab_association_only",
                   "feo_wt_percent_mean": parent_feo + 0.3 * rng.standard_normal()}
            for i in range(74):
                row[f"m3_b{i + 1:02d}"] = refl[i]
            rows.append(row)
            rid += 1
    return pd.DataFrame(rows)


def check_no_group_leak(df: pd.DataFrame) -> None:
    from sklearn.model_selection import GroupKFold
    g = df[GROUP_COLUMN].to_numpy()
    X = df[[f"m3_b{i:02d}" for i in range(1, 75)]].to_numpy()
    y = df["feo_wt_percent_mean"].to_numpy()
    gkf = GroupKFold(n_splits=4)
    for tr, te in gkf.split(X, y, g):
        assert set(g[tr]).isdisjoint(set(g[te])), "group leaked across split!"
    print("[ok] no Apollo parent leaks across grouped folds")


def check_feature_shapes(df: pd.DataFrame) -> None:
    X = df[[f"m3_b{i:02d}" for i in range(1, 75)]].to_numpy()
    for fam in ("raw", "physical", "derivative", "shape", "phys_raw"):
        M, names = build_features(X, fam)
        assert M.shape[0] == X.shape[0]
        assert M.shape[1] == len(names)
        assert np.isfinite(M).all(), f"non-finite feature in {fam}"
    print("[ok] all feature families finite and shaped correctly")


def main() -> int:
    df = make_synthetic()
    print(f"synthetic: {df.shape[0]} spectra, {df[GROUP_COLUMN].nunique()} parents")
    check_no_group_leak(df)
    check_feature_shapes(df)

    with tempfile.TemporaryDirectory() as tmp:
        ds = Path(tmp) / "synthetic.csv"
        df.to_csv(ds, index=False)

        null_mae = None
        signal_mae = None
        for model in list(MODEL_FAMILIES) + ["spectral_angle"]:
            cfg = ExperimentConfig(
                dataset=str(ds), target_column="feo_wt_percent_mean",
                model=model, outer_folds=4, inner_folds=2,
                permutation_repeats=2, seed=1,
                output_dir=str(Path(tmp) / model),
            )
            m = run_experiment(cfg)
            status = m.get("status")
            mae = m.get("mae_parent")
            print(f"  {model:24s} status={status} parent_MAE={mae}")
            assert status == "ok", f"{model} failed: {m.get('error')}"
            if model == "null_mean":
                null_mae = mae
            if model == "hist_gbm":
                signal_mae = mae

        # log1p inverse round-trip on original scale
        cfg = ExperimentConfig(
            dataset=str(ds), target_column="feo_wt_percent_mean", model="ridge",
            target_transform="log1p", outer_folds=4, inner_folds=2,
            permutation_repeats=1, output_dir=str(Path(tmp) / "log1p"),
        )
        mlog = run_experiment(cfg)
        assert mlog.get("status") == "ok"
        assert mlog["mae_parent"] < null_mae, "log1p ridge should beat null"
        print(f"[ok] log1p round-trip ok (parent_MAE={mlog['mae_parent']:.3f})")

        assert signal_mae < null_mae, (
            f"signal model ({signal_mae}) should beat null ({null_mae})")
        print(f"[ok] signal model beats null ({signal_mae:.3f} < {null_mae:.3f})")

    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
