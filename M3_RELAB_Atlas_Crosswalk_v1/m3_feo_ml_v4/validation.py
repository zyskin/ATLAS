"""v4 validation stage: corrected permutation importance, randomized-fold
stability repeats, ExtraTrees feature ablation, and Is/FeO transformed GP.

Writes everything under ``data/ml_experiments_v4_validation/`` and never touches
the original ``data/ml_experiments_v4/`` artefacts.  Only a shortlist of models
is re-run (never the full matrix).

Stages (``--stages`` to subset): perm, repeats, ablation, isfeo_gp.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .trainer import ExperimentConfig, run_experiment

warnings.filterwarnings("ignore")

DATASET = "data/ml_combined_v2/m3_feo_isfeo_ground_truth_v2.csv"

SCEN = {
    "feo_combined": dict(target="feo_wt_percent_mean", outer=5),
    "feo_direct": dict(target="feo_direct_wt_percent_mean", outer=5),
    "isfeo": dict(target="is_feo_mean", outer=4),
}

# (model, feature_family, target_transform, label)
PERM_SHORTLIST = {
    "feo_combined": [
        ("extra_trees", "phys_raw", "none", "extra_trees_phys_raw"),
        ("kernel_ridge_rbf", "raw", "none", "kernel_ridge_rbf_raw"),
        ("bayesian_ridge_spline", "physical", "none", "bayesian_ridge_spline_physical"),
    ],
    "feo_direct": [
        ("kernel_ridge_rbf", "raw", "none", "kernel_ridge_rbf_raw"),
    ],
    "isfeo": [
        ("gaussian_process", "raw", "none", "gaussian_process_raw"),
        ("extra_trees", "phys_raw", "none", "extra_trees_phys_raw"),
    ],
}

REPEAT_SHORTLIST = {
    "feo_combined": ([("extra_trees", "phys_raw", "none", "extra_trees_phys_raw"),
                      ("kernel_ridge_rbf", "raw", "none", "kernel_ridge_rbf_raw"),
                      ("bayesian_ridge_spline", "physical", "none", "bayesian_ridge_spline_physical")], 5),
    "feo_direct": ([("kernel_ridge_rbf", "raw", "none", "kernel_ridge_rbf_raw"),
                    ("extra_trees", "phys_raw", "none", "extra_trees_phys_raw")], 5),
    "isfeo": ([("gaussian_process", "raw", "none", "gaussian_process_raw"),
               ("extra_trees", "phys_raw", "none", "extra_trees_phys_raw"),
               ("kernel_ridge_rbf", "raw", "none", "kernel_ridge_rbf_raw")], 10),
}


def stage_perm(base: Path) -> None:
    """P1: corrected parent+measurement-level permutation importance (fixed folds)."""
    out = base / "permutation"
    rows = []
    for scen, items in PERM_SHORTLIST.items():
        for model, fam, tr, label in items:
            cfg = ExperimentConfig(
                dataset=DATASET, target_column=SCEN[scen]["target"], model=model,
                feature_family=fam, target_transform=tr, outer_folds=SCEN[scen]["outer"],
                inner_folds=3, seed=20261105, permutation_repeats=8,
                output_dir=str(out / scen / label), label=label,
                save_final_model=False,
            )
            m = run_experiment(cfg)
            pf = out / scen / label / "block_permutation_importance.csv"
            top = ""
            if pf.exists():
                pdf = pd.read_csv(pf).sort_values("mae_increase_parent", ascending=False)
                top = "; ".join(f"{r.block}={r.mae_increase_parent:.3f}"
                                for r in pdf.head(3).itertuples())
            rows.append({"scenario": scen, "label": label, "status": m.get("status"),
                         "parent_mae": m.get("mae_parent"), "top3_parent_blocks": top})
            print(f"[perm] {scen}/{label}: parent_MAE={m.get('mae_parent')} top3[{top}]")
    pd.DataFrame(rows).to_csv(out / "permutation_top_blocks.csv", index=False)


def stage_repeats(base: Path) -> None:
    """P3: repeated randomized grouped-CV stability (Is/FeO + FeO-direct first)."""
    out = base / "repeats"
    rows = []
    order = ["isfeo", "feo_direct", "feo_combined"]  # finish small/cheap first
    for scen in order:
        items, n_rep = REPEAT_SHORTLIST[scen]
        for model, fam, tr, label in items:
            for rep in range(n_rep):
                cfg = ExperimentConfig(
                    dataset=DATASET, target_column=SCEN[scen]["target"], model=model,
                    feature_family=fam, target_transform=tr, outer_folds=SCEN[scen]["outer"],
                    inner_folds=3, seed=20261105, permutation_repeats=0,
                    output_dir=str(out / scen / label / f"rep{rep:02d}"), label=label,
                    save_final_model=False, randomized_folds=True, fold_seed=1000 + rep,
                )
                m = run_experiment(cfg)
                rows.append({"scenario": scen, "label": label, "repeat": rep,
                             "fold_seed": 1000 + rep, "status": m.get("status"),
                             "parent_mae": m.get("mae_parent"),
                             "parent_rmse": m.get("rmse_parent"),
                             "parent_r2": m.get("r2_parent")})
                pd.DataFrame(rows).to_csv(out / "repeats_all.csv", index=False)
            sub = [r for r in rows if r["scenario"] == scen and r["label"] == label
                   and r["parent_mae"] is not None]
            maes = np.array([r["parent_mae"] for r in sub], dtype=float)
            print(f"[repeats] {scen}/{label}: n={len(maes)} parent_MAE "
                  f"mean={maes.mean():.3f} median={np.median(maes):.3f} "
                  f"std={maes.std(ddof=1):.3f} range=[{maes.min():.3f},{maes.max():.3f}]")
    # summary
    df = pd.read_csv(out / "repeats_all.csv")
    summ = (df.groupby(["scenario", "label"])["parent_mae"]
            .agg(["count", "mean", "median", "std", "min", "max"]).reset_index())
    summ.to_csv(out / "repeats_summary.csv", index=False)
    print("\n=== repeats summary ===")
    print(summ.to_string(index=False))


def stage_ablation(base: Path) -> None:
    """P4: ExtraTrees raw vs physical vs phys_raw on identical folds/tuning."""
    out = base / "ablation"
    rows = []
    for scen in ["feo_combined", "isfeo"]:
        for fam in ["raw", "physical", "phys_raw"]:
            label = f"extra_trees_{fam}"
            cfg = ExperimentConfig(
                dataset=DATASET, target_column=SCEN[scen]["target"], model="extra_trees",
                feature_family=fam, target_transform="none", outer_folds=SCEN[scen]["outer"],
                inner_folds=3, seed=20261105, permutation_repeats=0,
                output_dir=str(out / scen / label), label=label, save_final_model=False,
            )
            m = run_experiment(cfg)
            rows.append({"scenario": scen, "feature_family": fam, "status": m.get("status"),
                         "parent_mae": m.get("mae_parent"), "parent_rmse": m.get("rmse_parent"),
                         "parent_r2": m.get("r2_parent")})
            print(f"[ablation] {scen} extra_trees {fam}: parent_MAE={m.get('mae_parent')}")
    pd.DataFrame(rows).to_csv(out / "extra_trees_feature_ablation.csv", index=False)


def stage_isfeo_gp(base: Path) -> None:
    """P5: Is/FeO GP raw-target vs log1p-target (inverse inside folds, no clipping)."""
    out = base / "isfeo_gp"
    rows = []
    for tr, label in [("none", "gaussian_process_raw"), ("log1p", "gaussian_process_log1p")]:
        cfg = ExperimentConfig(
            dataset=DATASET, target_column="is_feo_mean", model="gaussian_process",
            feature_family="raw", target_transform=tr, outer_folds=4, inner_folds=3,
            seed=20261105, permutation_repeats=0, output_dir=str(out / label),
            label=label, save_final_model=True,
        )
        m = run_experiment(cfg)
        oof = pd.read_csv(out / label / "oof_predictions.csv").dropna(subset=["oof_prediction"])
        pm = oof.groupby("apollo_parent")[["target", "oof_prediction"]].mean()
        pm["abs_err"] = (pm["oof_prediction"] - pm["target"]).abs()
        worst = pm.sort_values("abs_err", ascending=False).head(3)
        rows.append({
            "label": label, "target_transform": tr,
            "parent_mae": m.get("mae_parent"), "parent_rmse": m.get("rmse_parent"),
            "parent_r2": m.get("r2_parent"),
            "negative_pred_count": int((oof["oof_prediction"] < 0).sum()),
            "out_of_range_count": int(((oof["oof_prediction"] < oof["target"].min()) |
                                       (oof["oof_prediction"] > oof["target"].max())).sum()),
            "pred_min": float(oof["oof_prediction"].min()),
            "pred_max": float(oof["oof_prediction"].max()),
            "worst_parents": "; ".join(f"{i}:t={r.target:.0f},p={r.oof_prediction:.1f}"
                                       for i, r in worst.iterrows()),
        })
        print(f"[isfeo_gp] {label}: MAE={m.get('mae_parent'):.3f} "
              f"neg={rows[-1]['negative_pred_count']} oor={rows[-1]['out_of_range_count']} "
              f"predrange=[{rows[-1]['pred_min']:.1f},{rows[-1]['pred_max']:.1f}]")
    pd.DataFrame(rows).to_csv(out / "isfeo_gp_raw_vs_log.csv", index=False)

    # repeated randomized stability for both variants
    rep_rows = []
    for tr, label in [("none", "gaussian_process_raw"), ("log1p", "gaussian_process_log1p")]:
        for rep in range(10):
            cfg = ExperimentConfig(
                dataset=DATASET, target_column="is_feo_mean", model="gaussian_process",
                feature_family="raw", target_transform=tr, outer_folds=4, inner_folds=3,
                seed=20261105, permutation_repeats=0,
                output_dir=str(out / "stability" / label / f"rep{rep:02d}"), label=label,
                save_final_model=False, randomized_folds=True, fold_seed=2000 + rep,
            )
            m = run_experiment(cfg)
            rep_rows.append({"label": label, "repeat": rep, "parent_mae": m.get("mae_parent")})
    rdf = pd.DataFrame(rep_rows)
    rsumm = rdf.groupby("label")["parent_mae"].agg(["count", "mean", "median", "std", "min", "max"]).reset_index()
    rdf.to_csv(out / "isfeo_gp_stability_all.csv", index=False)
    rsumm.to_csv(out / "isfeo_gp_stability_summary.csv", index=False)
    print("\n=== Is/FeO GP stability (randomized repeats) ===")
    print(rsumm.to_string(index=False))


STAGES = {"perm": stage_perm, "repeats": stage_repeats,
          "ablation": stage_ablation, "isfeo_gp": stage_isfeo_gp}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-dir", default="data/ml_experiments_v4_validation")
    ap.add_argument("--stages", nargs="+", default=list(STAGES), choices=list(STAGES))
    a = ap.parse_args()
    base = Path(a.base_dir)
    base.mkdir(parents=True, exist_ok=True)
    for s in a.stages:
        print(f"\n########## stage: {s} ##########")
        STAGES[s](base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
