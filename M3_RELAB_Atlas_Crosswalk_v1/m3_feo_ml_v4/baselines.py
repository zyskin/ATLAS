"""Fixed (published, unmodified) formula and trivial baselines, scored on the
same parents as the ML experiments.

These are *not* re-fitted: the Lucey/Kumar columns already stored in the dataset
are the published-formula predictions.  They are scored here only as reference
comparators on matched Apollo parents (with coverage reported, since some
formulas are undefined for part of the sample).  A parent-mean-of-target
"global mean" null predictor is also produced via leave-one-parent-out so it is
honest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

GROUP = "i_atlas_sample"

FEO_FORMULA_COLUMNS = [
    "lucey_standard_feo_principal",
    "lucey_standard_feo_atan2_2pi",
    "lucey_origin_changed_feo_principal",
    "lucey_origin_changed_feo_atan2_2pi",
    "kumar_feo_linear_principal",
    "kumar_feo_power_principal",
    "kumar_feo_linear_atan2_2pi",
    "kumar_feo_power_atan2_2pi",
]


def _parent_metrics(df: pd.DataFrame, target: str, pred: str) -> dict:
    sub = df.dropna(subset=[target, pred])
    if sub.empty:
        return {}
    pm = sub.groupby(GROUP)[[target, pred]].mean()
    err = pm[pred] - pm[target]
    ss_res = float((err ** 2).sum())
    ss_tot = float(((pm[target] - pm[target].mean()) ** 2).sum())
    return {
        "n_parents": int(pm.shape[0]),
        "n_spectra": int(sub.shape[0]),
        "mae_parent": float(err.abs().mean()),
        "rmse_parent": float(np.sqrt((err ** 2).mean())),
        "bias_parent": float(err.mean()),
        "r2_parent": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def leave_one_parent_out_mean(df: pd.DataFrame, target: str, folds: int = 5) -> dict:
    sub = df.dropna(subset=[target]).reset_index(drop=True)
    g = sub[GROUP].to_numpy()
    y = sub[target].to_numpy(dtype=float)
    pred = np.full(y.shape, np.nan)
    n = min(folds, np.unique(g).size)
    gkf = GroupKFold(n_splits=max(2, n))
    for tr, te in gkf.split(sub, y, g):
        pred[te] = y[tr].mean()
    tmp = pd.DataFrame({GROUP: g, target: y, "pred": pred})
    return _parent_metrics(tmp, target, "pred")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--target-column", default="feo_wt_percent_mean")
    p.add_argument("--subset", default="all", choices=["all", "exact_only"])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--folds", type=int, default=5)
    a = p.parse_args()

    df = pd.read_csv(a.dataset)
    if a.subset == "exact_only":
        df = df[df["feo_target_scope"].isin(
            {"exact_relab_association_only", "mixed_exact_and_parent"})]
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    # published fixed formulas (FeO only)
    if "feo" in a.target_column.lower():
        for col in FEO_FORMULA_COLUMNS:
            if col in df.columns:
                m = _parent_metrics(df, a.target_column, col)
                if m:
                    rows.append({"baseline": col, "kind": "fixed_published_formula", **m})
    # trivial null predictor (grouped leave-one-parent-out mean)
    m = leave_one_parent_out_mean(df, a.target_column, a.folds)
    if m:
        rows.append({"baseline": "grouped_mean_null", "kind": "null", **m})

    res = pd.DataFrame(rows).sort_values("mae_parent")
    res.to_csv(out / "fixed_formula_baselines.csv", index=False)
    print(json.dumps({"target": a.target_column, "subset": a.subset,
                      "baselines": rows}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
