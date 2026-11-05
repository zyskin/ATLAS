"""Generate m3_feo_ml_v4/VALIDATION_RESULTS.md from the saved validation-stage
artefacts (bootstrap, permutation, repeats, ablation, isfeo_gp).  All numbers
come from the executed-run CSVs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def rd(x, n=3):
    try:
        return f"{float(x):.{n}f}"
    except (TypeError, ValueError):
        return str(x)


def md_table(df: pd.DataFrame, cols: list[str], headers: list[str], nd=3) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r.get(c)
            cells.append(rd(v, nd) if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-dir", default="data/ml_experiments_v4_validation")
    ap.add_argument("--out", default="m3_feo_ml_v4/VALIDATION_RESULTS.md")
    a = ap.parse_args()
    v = Path(a.val_dir)
    M: list[str] = []
    def w(s=""): M.append(s)

    w("# v4 validation results — robustness of the reported FeO/Is-FeO improvements")
    w()
    w("Validation stage for branch `claude/apollo-feo-isfeo-v4`. All metrics are "
      "Apollo-parent-grouped. Outputs live under `data/ml_experiments_v4_validation/` "
      "and the original `data/ml_experiments_v4/` results are unchanged.")
    w()

    # ---- P1 permutation ----
    w("## 1. Corrected permutation importance (parent-level)")
    w()
    w("The earlier block-permutation metric used **measurement-level** MAE while "
      "the plots/README described **parent-level** MAE. `trainer.py` now computes "
      "both; parent-level aggregates predictions and targets by Apollo parent "
      "*after* each permutation. Recomputed for the top models (8 permutation "
      "repeats/fold, fixed deterministic folds).")
    w()
    ptb = v / "permutation" / "permutation_top_blocks.csv"
    if ptb.exists():
        df = pd.read_csv(ptb)
        w(md_table(df, ["scenario", "label", "parent_mae", "top3_parent_blocks"],
                   ["scenario", "model", "parent MAE", "top-3 parent-level blocks (ΔMAE)"]))
        w()
    w("_Interpretation note: whether bands 17–24 (≈930–1050 nm, the 1 µm ferrous "
      "band) still dominate FeO under parent-level scoring is read directly from "
      "the top-3 blocks above._")
    w()

    # ---- P2 bootstrap ----
    w("## 2. Apollo-parent bootstrap CIs and paired ΔMAE")
    w()
    w("Cluster bootstrap over Apollo parents (10,000 resamples) on the saved "
      "out-of-fold parent-mean predictions; no retraining. v3 saved only aggregate "
      "metrics (no per-parent predictions), so it appears only as an **unpaired** "
      "reference.")
    w()
    pm = v / "bootstrap" / "per_model_ci.csv"
    if pm.exists():
        df = pd.read_csv(pm)
        w("**Per-model parent-MAE/RMSE 95% CI:**")
        w()
        rows = []
        for _, r in df.iterrows():
            rows.append({"scenario": r["scenario"], "model": r["model"],
                         "mae": f"{rd(r['mae'])} [{rd(r['mae_ci_lo'])}, {rd(r['mae_ci_hi'])}]",
                         "rmse": f"{rd(r['rmse'])} [{rd(r['rmse_ci_lo'])}, {rd(r['rmse_ci_hi'])}]",
                         "bias": f"{rd(r['bias'])} [{rd(r['bias_ci_lo'])}, {rd(r['bias_ci_hi'])}]"})
        w(md_table(pd.DataFrame(rows), ["scenario", "model", "mae", "rmse", "bias"],
                   ["scenario", "model", "parent MAE [95% CI]", "parent RMSE [95% CI]", "bias [95% CI]"]))
        w()
    pd_ = v / "bootstrap" / "paired_delta_mae.csv"
    if pd_.exists():
        df = pd.read_csv(pd_)
        w("**Paired ΔMAE (model_A − model_B; <0 ⇒ A better; identical resampled parents):**")
        w()
        rows = []
        for _, r in df.iterrows():
            rows.append({"scenario": r["scenario"], "pair": f"{r['model_A']} − {r['model_B']}",
                         "dmae": f"{rd(r['obs_delta_mae_A_minus_B'])} [{rd(r['delta_ci_lo'])}, {rd(r['delta_ci_hi'])}]",
                         "winA": rd(r["frac_A_wins"], 3),
                         "sig": "yes" if r["ci_excludes_zero"] else "no"})
        w(md_table(pd.DataFrame(rows), ["scenario", "pair", "dmae", "winA", "sig"],
                   ["scenario", "A − B", "ΔMAE [95% CI]", "P(A wins)", "CI excl. 0?"]))
        w()
    ref = v / "bootstrap" / "v3_unpaired_reference.csv"
    if ref.exists():
        df = pd.read_csv(ref)
        w("**v3 unpaired reference (point estimates):**")
        w()
        w(md_table(df, ["scenario", "v3_method", "v3_mae_point", "v3_n_parents"],
                   ["scenario", "v3 method", "v3 parent MAE", "n_parents"]))
        w()

    # ---- P3 repeats ----
    w("## 3. Repeated randomized grouped-CV stability")
    w()
    w("`RandomizedGroupKFold` shuffles whole Apollo parents into folds with a "
      "seeded partition (ordinary `GroupKFold` gives one deterministic, "
      "seed-independent split). Inner model selection stays parent-grouped.")
    w()
    rs = v / "repeats" / "repeats_summary.csv"
    if rs.exists():
        df = pd.read_csv(rs)
        w(md_table(df, ["scenario", "label", "count", "mean", "median", "std", "min", "max"],
                   ["scenario", "model", "repeats", "mean MAE", "median", "std", "min", "max"]))
        w()

    # ---- P4 ablation ----
    w("## 4. ExtraTrees feature ablation (identical folds + tuning)")
    w()
    w("Isolates the feature-engineering contribution from model/tuning changes: "
      "the same ExtraTrees machinery and grouped folds, varying only the feature "
      "family.")
    w()
    ab = v / "ablation" / "extra_trees_feature_ablation.csv"
    if ab.exists():
        df = pd.read_csv(ab)
        w(md_table(df, ["scenario", "feature_family", "parent_mae", "parent_rmse", "parent_r2"],
                   ["scenario", "feature family", "parent MAE", "parent RMSE", "parent R²"]))
        w()
    w("_The stored matrix also has HistGradientBoosting raw (FeO combined MAE 2.795) "
      "vs phys_raw (2.376), an independent indication of a feature effect._")
    w()

    # ---- P5 isfeo gp ----
    w("## 5. Is/FeO GP: raw target vs log1p target")
    w()
    w("GP with `log1p` target transform fitted and inverted **inside** each fold "
      "(no clipping), vs the raw-target GP.")
    w()
    rv = v / "isfeo_gp" / "isfeo_gp_raw_vs_log.csv"
    if rv.exists():
        df = pd.read_csv(rv)
        w(md_table(df, ["label", "parent_mae", "parent_rmse", "parent_r2",
                        "negative_pred_count", "out_of_range_count", "pred_min", "pred_max"],
                   ["variant", "parent MAE", "RMSE", "R²", "neg preds", "out-of-range",
                    "pred min", "pred max"]))
        w()
    ss = v / "isfeo_gp" / "isfeo_gp_stability_summary.csv"
    if ss.exists():
        df = pd.read_csv(ss)
        w("**Randomized-repeat stability (10 repeats):**")
        w()
        w(md_table(df, ["label", "count", "mean", "median", "std", "min", "max"],
                   ["variant", "repeats", "mean MAE", "median", "std", "min", "max"]))
        w()

    Path(a.out).write_text("\n".join(M) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
