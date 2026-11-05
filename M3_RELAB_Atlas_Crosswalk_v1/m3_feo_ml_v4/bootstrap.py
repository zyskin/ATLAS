"""P2: Apollo-parent cluster bootstrap confidence intervals and paired ΔMAE.

Uses the already-saved out-of-fold predictions (deterministic GroupKFold, so all
v4 models share an identical parent partition) aggregated to Apollo-parent means.
No models are retrained.

For each scenario it resamples whole Apollo parents with replacement B times
(default 10000) and reports, per model, 95% percentile CIs for parent MAE, RMSE
and bias; and for each model pair, the paired ΔMAE distribution on *identical*
resampled parents plus the fraction of resamples in which each model wins.

Note: the stored v3 models saved only aggregate `cv_metrics.csv`, not per-parent
predictions, so v3 cannot enter the *paired* bootstrap. v3 point estimates are
reported as an unpaired reference against each v4 model's MAE CI.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

RUN = "data/ml_experiments_v4"
V3_METRICS = {
    "feo_combined": "data/ml_interpretable_v3/results_combined_quick/cv_metrics.csv",
    "feo_direct": "data/ml_interpretable_v3/results_direct_only_quick/cv_metrics.csv",
}

SCEN_MODELS = {
    "feo_combined": ["extra_trees_phys_raw", "kernel_ridge_rbf_raw",
                     "bayesian_ridge_spline_physical"],
    "feo_direct": ["kernel_ridge_rbf_raw", "extra_trees_phys_raw"],
    "isfeo": ["gaussian_process_raw", "extra_trees_phys_raw",
              "kernel_ridge_rbf_raw", "null_mean"],
}


def parent_frame(scenario: str, labels: list[str]) -> pd.DataFrame:
    """Parent-mean target + per-model parent-mean prediction, aligned by parent."""
    base = None
    for label in labels:
        p = Path(RUN) / scenario / label / "oof_predictions.csv"
        if not p.exists():
            print(f"  (missing {p}; skipping {label})")
            continue
        df = pd.read_csv(p).dropna(subset=["oof_prediction"])
        pm = df.groupby("apollo_parent").agg(target=("target", "mean"),
                                             pred=("oof_prediction", "mean"))
        pm = pm.rename(columns={"pred": f"pred__{label}"})
        if base is None:
            base = pm
        else:
            base = base.join(pm[[f"pred__{label}"]], how="inner")
    return base


def ci(a, lo=2.5, hi=97.5):
    return float(np.percentile(a, lo)), float(np.percentile(a, hi))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default=RUN)
    ap.add_argument("--out-dir", default="data/ml_experiments_v4_validation/bootstrap")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261105)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    per_model_rows = []
    paired_rows = []
    v3_ref_rows = []

    for scenario, labels in SCEN_MODELS.items():
        pf = parent_frame(scenario, labels)
        if pf is None or pf.empty:
            continue
        present = [l for l in labels if f"pred__{l}" in pf.columns]
        n = pf.shape[0]
        target = pf["target"].to_numpy()
        # per-parent errors per model
        err = {l: pf[f"pred__{l}"].to_numpy() - target for l in present}

        # shared bootstrap resamples of parents
        idx = rng.integers(0, n, size=(a.n_boot, n))

        boot_mae = {}
        for l in present:
            e = err[l]
            ae = np.abs(e)
            se = e ** 2
            mae_b = ae[idx].mean(axis=1)
            rmse_b = np.sqrt(se[idx].mean(axis=1))
            bias_b = e[idx].mean(axis=1)
            boot_mae[l] = mae_b
            mlo, mhi = ci(mae_b)
            rlo, rhi = ci(rmse_b)
            blo, bhi = ci(bias_b)
            per_model_rows.append({
                "scenario": scenario, "model": l, "n_parents": n,
                "mae": float(np.mean(ae)), "mae_ci_lo": mlo, "mae_ci_hi": mhi,
                "rmse": float(np.sqrt(np.mean(se))), "rmse_ci_lo": rlo, "rmse_ci_hi": rhi,
                "bias": float(np.mean(e)), "bias_ci_lo": blo, "bias_ci_hi": bhi,
            })

        # paired ΔMAE for every pair (identical resampled parents)
        for A, B in combinations(present, 2):
            d = boot_mae[A] - boot_mae[B]          # <0 => A better
            dlo, dhi = ci(d)
            obs = float(np.mean(np.abs(err[A])) - np.mean(np.abs(err[B])))
            paired_rows.append({
                "scenario": scenario, "model_A": A, "model_B": B,
                "obs_delta_mae_A_minus_B": obs,
                "delta_ci_lo": dlo, "delta_ci_hi": dhi,
                "frac_A_wins": float(np.mean(d < 0)),
                "frac_B_wins": float(np.mean(d > 0)),
                "ci_excludes_zero": bool(dlo > 0 or dhi < 0),
            })

        # v3 unpaired reference
        v3p = V3_METRICS.get(scenario)
        if v3p and Path(v3p).exists():
            v3 = pd.read_csv(v3p)
            v3 = v3[v3["evaluation_unit"] == "apollo_parent_mean"]
            for name in ["gaussian_process", "gradient_boosting", "extra_trees"]:
                row = v3[v3["method"] == name]
                if not row.empty:
                    v3_ref_rows.append({"scenario": scenario, "v3_method": name,
                                        "v3_mae_point": float(row.iloc[0]["mae"]),
                                        "v3_n_parents": int(row.iloc[0]["n"]),
                                        "note": "unpaired: v3 per-parent predictions not saved"})

    pd.DataFrame(per_model_rows).to_csv(out / "per_model_ci.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(out / "paired_delta_mae.csv", index=False)
    pd.DataFrame(v3_ref_rows).to_csv(out / "v3_unpaired_reference.csv", index=False)

    print("=== per-model parent-MAE 95% CI ===")
    print(pd.DataFrame(per_model_rows)[
        ["scenario", "model", "mae", "mae_ci_lo", "mae_ci_hi"]].to_string(index=False))
    print("\n=== paired ΔMAE (A-B; <0 => A better) ===")
    print(pd.DataFrame(paired_rows)[
        ["scenario", "model_A", "model_B", "obs_delta_mae_A_minus_B",
         "delta_ci_lo", "delta_ci_hi", "frac_A_wins", "ci_excludes_zero"]].to_string(index=False))
    print("\n=== v3 unpaired reference ===")
    print(pd.DataFrame(v3_ref_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
