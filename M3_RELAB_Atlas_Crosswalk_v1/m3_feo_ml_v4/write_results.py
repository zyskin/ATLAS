"""Generate RESULTS.md from the saved v4 comparison/robustness outputs.

Everything printed comes from the executed-run CSV/JSON artefacts, so numbers in
the markdown match the validation outputs exactly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

SCEN_TITLE = {
    "feo_combined": "FeO combined (`feo_wt_percent_mean`, 107 parents)",
    "feo_direct": "FeO strict direct-only (`feo_direct_wt_percent_mean`, 33 parents)",
    "isfeo": "Is/FeO maturity (`is_feo_mean`, 29 parents)",
}


def fmt(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def leaderboard_md(base: Path, scen: str) -> str:
    path = base / f"comparison_v3_v4_{scen}.csv"
    if not path.exists():
        return f"*(no leaderboard for {scen})*\n"
    lb = pd.read_csv(path).sort_values("mae_parent")
    lines = ["| rank | method | source | parent MAE | parent RMSE | parent R² | n_parents |",
             "|---:|---|---|---:|---:|---:|---:|"]
    for i, (_, r) in enumerate(lb.iterrows(), 1):
        lines.append(
            f"| {i} | {r['method']} | {r['source']} | {fmt(r['mae_parent'])} | "
            f"{fmt(r['rmse_parent'])} | {fmt(r['r2_parent'])} | {int(r['n_parents'])} |")
    return "\n".join(lines) + "\n"


def best_ml(base: Path, scen: str):
    path = base / f"comparison_v3_v4_{scen}.csv"
    lb = pd.read_csv(path)
    ml = lb[lb["source"].astype(str).str.startswith("v4_ml")].sort_values("mae_parent")
    return ml.iloc[0] if not ml.empty else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="data/ml_experiments_v4")
    ap.add_argument("--out", default="m3_feo_ml_v4/RESULTS.md")
    a = ap.parse_args()
    base = Path(a.base_dir)

    md = ["# v4 results — Apollo/RELAB FeO & Is/FeO (parent-grouped CV)",
          "",
          "All metrics are **out-of-fold, Apollo-parent-grouped** (primary unit = "
          "Apollo-parent mean). `v4_ml_model` = new models from this package; "
          "`v4_fixed_published_formula` / `v4_null` = baselines scored on matched "
          "parents; `v3_stored` = metrics from `data/ml_interpretable_v3`. "
          "Fixed-formula rows with `n_parents < total` have partial coverage.",
          ""]

    for scen in ["feo_combined", "feo_direct", "isfeo"]:
        md.append(f"## {SCEN_TITLE.get(scen, scen)}\n")
        b = best_ml(base, scen)
        if b is not None:
            md.append(
                f"**Best new model: `{b['method']}`** — parent MAE "
                f"{fmt(b['mae_parent'])}, RMSE {fmt(b['rmse_parent'])}, "
                f"R² {fmt(b['r2_parent'])} (n={int(b['n_parents'])} parents).\n")
        md.append(leaderboard_md(base, scen))

        rj = base / f"robustness_{scen}.json"
        if rj.exists():
            rep = json.loads(rj.read_text())
            md.append(f"\n**Robustness ({rep.get('best_model')}):**\n")
            neg = rep.get("negative_predictions", {})
            md.append(f"- Negative predictions: {neg.get('count')} "
                      f"({fmt(100 * (neg.get('fraction') or 0), 1)}% of spectra).")
            ps = rep.get("perturbation_sensitivity", {})
            if "mean_parent_pred_shift_multiplicative" in ps:
                md.append(
                    f"- Reflectance perturbation: +2% multiplicative noise shifts "
                    f"parent-mean prediction by "
                    f"{fmt(ps['mean_parent_pred_shift_multiplicative'])} "
                    f"(units of target); +0.005 additive by "
                    f"{fmt(ps['mean_parent_pred_shift_additive'])}.")
            sq = rep.get("source_quality_breakdown", {})
            if sq:
                parts = [f"{k}: MAE {fmt(v['mae_parent'])} (n={v['n_parents']})"
                         for k, v in sq.items()]
                md.append("- Source-quality breakdown — " + "; ".join(parts) + ".")
            oor = rep.get("isfeo_out_of_observed_range")
            if oor:
                md.append(
                    f"- Is/FeO predictions outside observed range "
                    f"[{fmt(oor['observed_min'],1)}, {fmt(oor['observed_max'],1)}]: "
                    f"{oor['count_outside']} (pred range "
                    f"[{fmt(oor['pred_min'],1)}, {fmt(oor['pred_max'],1)}]).")
            worst = rep.get("worst_parents", [])
            if worst:
                w = worst[0]
                md.append(f"- Worst parent: {w.get('apollo_parent')} "
                          f"(target {fmt(w.get('target'))}, pred "
                          f"{fmt(w.get('oof_prediction'))}, |err| {fmt(w.get('abs_error'))}).")
            md.append("")

    Path(a.out).write_text("\n".join(md) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
