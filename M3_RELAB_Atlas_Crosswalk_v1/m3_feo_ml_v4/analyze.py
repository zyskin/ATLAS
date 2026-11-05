"""Aggregate v4 results, merge with stored v3 baselines, and make GitHub-sized
plots and comparison tables.

Produces, under --base-dir:
  * comparison_v3_v4_<scenario>.csv : unified parent-level leaderboard (v4 models,
    v4 fixed/null baselines, and the stored v3 parent-mean metrics), with
    n_parents coverage so partial-coverage formulas are not over-claimed.
  * figures/: predicted-vs-actual scatter for the best v4 model, block
    permutation-importance bars, and linear standardized-coefficient bars.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

V3_METRICS = {
    "feo_combined": "data/ml_interpretable_v3/results_combined_quick/cv_metrics.csv",
    "feo_direct": "data/ml_interpretable_v3/results_direct_only_quick/cv_metrics.csv",
}


def load_v4_comparison(base: Path) -> pd.DataFrame:
    path = base / "comparison_all.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def build_scenario_leaderboard(base: Path, scenario: str, v4: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sub = v4[v4["scenario"] == scenario]
    for _, r in sub.iterrows():
        rows.append({
            "method": r.get("label"),
            "source": "v4_" + str(r.get("kind", "")),
            "mae_parent": r.get("mae_parent"),
            "rmse_parent": r.get("rmse_parent"),
            "r2_parent": r.get("r2_parent"),
            "bias_parent": r.get("bias_parent"),
            "n_parents": r.get("n_parents"),
            "feature_family": r.get("feature_family", ""),
            "target_transform": r.get("target_transform", ""),
        })
    # stored v3 parent-mean metrics
    v3_path = V3_METRICS.get(scenario)
    if v3_path and Path(v3_path).exists():
        v3 = pd.read_csv(v3_path)
        v3 = v3[v3["evaluation_unit"] == "apollo_parent_mean"]
        for _, r in v3.iterrows():
            rows.append({
                "method": r["method"],
                "source": "v3_stored",
                "mae_parent": r["mae"],
                "rmse_parent": r["rmse"],
                "r2_parent": r["r2"],
                "bias_parent": r["bias"],
                "n_parents": int(r["n"]),
                "feature_family": "",
                "target_transform": "",
            })
    lb = pd.DataFrame(rows).sort_values("mae_parent").reset_index(drop=True)
    return lb


def plot_pred_vs_actual(scen_dir: Path, scenario: str, out: Path, best_label: str) -> None:
    pred_path = scen_dir / best_label / "oof_predictions.csv"
    if not pred_path.exists():
        return
    df = pd.read_csv(pred_path).dropna(subset=["oof_prediction"])
    pm = df.groupby("apollo_parent")[["target", "oof_prediction"]].mean()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(pm["target"], pm["oof_prediction"], s=22, alpha=0.75, edgecolor="k", linewidth=0.3)
    lo = float(min(pm.min().min(), 0))
    hi = float(pm.max().max()) * 1.05
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlabel("Apollo-parent mean target")
    ax.set_ylabel("Out-of-fold parent-mean prediction")
    ax.set_title(f"{scenario}: {best_label}\nparent-grouped OOF")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out / f"pred_vs_actual_{scenario}.png", dpi=110)
    plt.close(fig)


def plot_permutation(scen_dir: Path, scenario: str, out: Path, best_label: str) -> None:
    perm_path = scen_dir / best_label / "block_permutation_importance.csv"
    if not perm_path.exists():
        return
    df = pd.read_csv(perm_path).sort_values("mae_increase", ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(df["block"], df["mae_increase"], xerr=df.get("mae_increase_fold_std"),
            color="#3a7ca5")
    ax.set_xlabel("Parent-MAE increase when block permuted (held-out)")
    ax.set_title(f"{scenario}: block permutation importance\n{best_label}")
    fig.tight_layout()
    fig.savefig(out / f"permutation_{scenario}.png", dpi=110)
    plt.close(fig)


def plot_coefficients(scen_dir: Path, scenario: str, out: Path) -> None:
    # pick a linear model dir that produced coefficients
    for cand in scen_dir.glob("*/linear_standardized_coefficients.csv"):
        df = pd.read_csv(cand).head(15).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = ["#c1432e" if v < 0 else "#2e7d32" for v in df["standardized_coef"]]
        ax.barh(df["feature"], df["standardized_coef"], color=colors)
        ax.set_xlabel("Standardized coefficient")
        ax.set_title(f"{scenario}: {cand.parent.name} standardized coefficients")
        fig.tight_layout()
        fig.savefig(out / f"coefficients_{scenario}_{cand.parent.name}.png", dpi=110)
        plt.close(fig)
        break


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-dir", default="data/ml_experiments_v4")
    p.add_argument("--scenarios", nargs="+", default=["feo_combined", "feo_direct", "isfeo"])
    a = p.parse_args()

    base = Path(a.base_dir)
    figs = base / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    v4 = load_v4_comparison(base)

    for scen in a.scenarios:
        lb = build_scenario_leaderboard(base, scen, v4)
        lb.to_csv(base / f"comparison_v3_v4_{scen}.csv", index=False)
        print(f"\n=== {scen} leaderboard (parent-grouped MAE) ===")
        print(lb.to_string(index=False))

        # best v4 ML model by parent MAE
        ml = lb[lb["source"].str.startswith("v4_ml")]
        if not ml.empty:
            best = ml.iloc[0]["method"]
            scen_dir = base / scen
            plot_pred_vs_actual(scen_dir, scen, figs, best)
            plot_permutation(scen_dir, scen, figs, best)
            plot_coefficients(scen_dir, scen, figs)
    print(f"\nfigures in {figs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
