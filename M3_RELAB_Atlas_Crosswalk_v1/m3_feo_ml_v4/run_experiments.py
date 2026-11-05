"""Run the full v4 experiment matrix for FeO (combined + direct-only) and Is/FeO.

Each experiment writes its own output subdirectory under --base-dir and appends
one row to the master comparison CSV *as soon as it finishes*, so a time-limited
session always retains completed work.  Fixed-formula and grouped-null baselines
are scored per scenario on the matched parents.

Usage:
    python -m m3_feo_ml_v4.run_experiments \
        --dataset data/ml_combined_v2/m3_feo_isfeo_ground_truth_v2.csv \
        --base-dir data/ml_experiments_v4 [--scenarios feo_combined feo_direct isfeo]
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import pandas as pd

from .baselines import FEO_FORMULA_COLUMNS, _parent_metrics, leave_one_parent_out_mean
from .trainer import EXACT_SCOPES, ExperimentConfig, run_experiment

warnings.filterwarnings("ignore")

# (model, feature_family or None=default, target_transform, label suffix)
FEO_MODELS = [
    ("huber", "physical", "none", "huber_physical"),
    ("bayesian_ridge", "physical", "none", "bayesian_ridge_physical"),
    ("huber_spline", "physical", "none", "huber_spline_physical"),
    ("bayesian_ridge_spline", "physical", "none", "bayesian_ridge_spline_physical"),
    ("ridge", "raw", "none", "ridge_raw"),
    ("hist_gbm", "phys_raw", "none", "hist_gbm_phys_raw"),
    ("hist_gbm", "raw", "none", "hist_gbm_raw"),
    ("kernel_ridge_rbf", "raw", "none", "kernel_ridge_rbf_raw"),
    ("gaussian_process", "raw", "none", "gaussian_process_raw"),
    ("extra_trees", "phys_raw", "none", "extra_trees_phys_raw"),
    ("spectral_angle", "raw", "none", "spectral_angle_modified"),
    ("null_mean", "physical", "none", "null_mean"),
]

ISFEO_MODELS = [
    ("ridge", "physical", "none", "ridge_physical"),
    ("ridge", "physical", "log1p", "ridge_physical_log1p"),
    ("huber", "physical", "none", "huber_physical"),
    ("huber", "physical", "log1p", "huber_physical_log1p"),
    ("bayesian_ridge", "physical", "none", "bayesian_ridge_physical"),
    ("extra_trees", "phys_raw", "none", "extra_trees_phys_raw"),
    ("hist_gbm", "phys_raw", "none", "hist_gbm_phys_raw"),
    ("kernel_ridge_rbf", "raw", "none", "kernel_ridge_rbf_raw"),
    ("kernel_ridge_rbf", "raw", "log1p", "kernel_ridge_rbf_raw_log1p"),
    ("gaussian_process", "raw", "none", "gaussian_process_raw"),
    ("null_mean", "physical", "none", "null_mean"),
]

SCENARIOS = {
    "feo_combined": dict(target_column="feo_wt_percent_mean", subset="all",
                         outer_folds=5, inner_folds=3, models=FEO_MODELS),
    "feo_direct": dict(target_column="feo_direct_wt_percent_mean", subset="all",
                       outer_folds=5, inner_folds=3, models=FEO_MODELS),
    "isfeo": dict(target_column="is_feo_mean", subset="all",
                  outer_folds=4, inner_folds=3, models=ISFEO_MODELS),
}


def score_baselines(dataset: str, target: str, scen_dir: Path, folds: int) -> list[dict]:
    df = pd.read_csv(dataset)
    rows = []
    if "feo" in target.lower() and "is_feo" not in target.lower():
        for col in FEO_FORMULA_COLUMNS:
            if col in df.columns:
                m = _parent_metrics(df, target, col)
                if m:
                    rows.append({"label": col, "kind": "fixed_published_formula", **m})
    m = leave_one_parent_out_mean(df, target, folds)
    if m:
        rows.append({"label": "grouped_mean_null_baseline", "kind": "null", **m})
    pd.DataFrame(rows).to_csv(scen_dir / "fixed_formula_baselines.csv", index=False)
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--base-dir", default="data/ml_experiments_v4")
    p.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    p.add_argument("--seed", type=int, default=20261105)
    p.add_argument("--permutation-repeats", type=int, default=4)
    a = p.parse_args()

    base = Path(a.base_dir)
    base.mkdir(parents=True, exist_ok=True)
    comparison_path = base / "comparison_all.csv"
    all_rows: list[dict] = []

    for scen in a.scenarios:
        spec = SCENARIOS[scen]
        scen_dir = base / scen
        scen_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n===== scenario: {scen} (target={spec['target_column']}) =====")

        base_rows = score_baselines(a.dataset, spec["target_column"], scen_dir,
                                    spec["outer_folds"])
        for b in base_rows:
            all_rows.append({"scenario": scen, "target_column": spec["target_column"],
                             **b})
            print(f"  [baseline] {b['label']:34s} parent_MAE={b.get('mae_parent'):.3f} "
                  f"(n_par={b.get('n_parents')})")

        for model, fam, transform, label in spec["models"]:
            cfg = ExperimentConfig(
                dataset=a.dataset, target_column=spec["target_column"],
                model=model, feature_family=fam, subset=spec["subset"],
                target_transform=transform, outer_folds=spec["outer_folds"],
                inner_folds=spec["inner_folds"], seed=a.seed,
                permutation_repeats=a.permutation_repeats,
                output_dir=str(scen_dir / label), label=label,
            )
            try:
                m = run_experiment(cfg)
            except Exception as exc:
                m = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                     "label": label, "model": model}
            row = {"scenario": scen, "kind": "ml_model", **m}
            all_rows.append(row)
            # incremental save after EVERY model
            pd.DataFrame(all_rows).to_csv(comparison_path, index=False)
            if m.get("status") == "ok":
                print(f"  [model] {label:34s} parent_MAE={m.get('mae_parent'):.3f} "
                      f"rmse={m.get('rmse_parent'):.3f} r2={m.get('r2_parent'):.3f} "
                      f"(n_par={m.get('n_parents')})")
            else:
                print(f"  [model] {label:34s} FAILED: {m.get('error')}")

    print(f"\nwrote {comparison_path}")
    print(json.dumps({"n_rows": len(all_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
