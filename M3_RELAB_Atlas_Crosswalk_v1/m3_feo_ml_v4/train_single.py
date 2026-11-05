"""CLI to run a single configurable v4 experiment.

Example:
    python -m m3_feo_ml_v4.train_single \
        --dataset data/ml_combined_v2/m3_feo_isfeo_ground_truth_v2.csv \
        --target-column feo_wt_percent_mean --model hist_gbm \
        --feature-family phys_raw --outer-folds 5 --inner-folds 3 \
        --output-dir data/ml_experiments_v4/feo_combined/hist_gbm
"""
from __future__ import annotations

import argparse
import json

from .trainer import ExperimentConfig, run_experiment


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--target-column", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--feature-family", default=None)
    p.add_argument("--subset", default="all", choices=["all", "exact_only"])
    p.add_argument("--target-transform", default="none", choices=["none", "log1p"])
    p.add_argument("--outer-folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=20261105)
    p.add_argument("--permutation-repeats", type=int, default=5)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--label", default=None)
    a = p.parse_args()

    cfg = ExperimentConfig(
        dataset=a.dataset, target_column=a.target_column, model=a.model,
        feature_family=a.feature_family, subset=a.subset,
        target_transform=a.target_transform, outer_folds=a.outer_folds,
        inner_folds=a.inner_folds, seed=a.seed,
        permutation_repeats=a.permutation_repeats, output_dir=a.output_dir,
        label=a.label,
    )
    metrics = run_experiment(cfg)
    print(json.dumps(metrics, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
