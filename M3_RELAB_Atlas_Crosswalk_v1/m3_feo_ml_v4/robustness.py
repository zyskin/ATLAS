"""Grouped-holdout robustness / stress tests for v4 models.

Operates on the out-of-fold predictions and the saved final models, needing no
orbital M3 data.  For each scenario it reports:

  * source-quality breakdown of OOF parent error (exact vs parent-level scope),
  * worst-predicted Apollo parents (largest |OOF error|),
  * out-of-range predictions: negative FeO, and Is/FeO predictions outside the
    observed training range or negative,
  * within-parent prediction spread (stability across replicate spectra),
  * reflectance-perturbation sensitivity of the final model: small additive and
    multiplicative noise applied to all bands, change in parent-mean prediction.

Writes robustness_<scenario>.json and a combined robustness_summary.csv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import build_features

BAND_COLS = [f"m3_b{i:02d}" for i in range(1, 75)]


def _predict_from_bands(bundle, bands: np.ndarray) -> np.ndarray:
    fam = bundle["config"]["feature_family"] or _default_family(bundle)
    X, _ = build_features(bands, fam)
    z = bundle["estimator"].predict(X)
    t = bundle.get("target_transform", "none")
    if t == "log1p":
        z = np.expm1(z)
    return np.asarray(z, dtype=np.float64)


def _default_family(bundle):
    from .models import DEFAULT_FEATURE_FAMILY
    return DEFAULT_FEATURE_FAMILY.get(bundle["config"]["model"], "raw")


def perturbation_sensitivity(bundle, dataset: str, target: str, seed=0,
                             add_sigma=0.005, mult_sigma=0.02, reps=20) -> dict:
    rng = np.random.default_rng(seed)
    df = pd.read_csv(dataset)
    df = df[df[target].notna()]
    if bundle["config"].get("subset") == "exact_only":
        df = df[df["feo_target_scope"].isin(
            {"exact_relab_association_only", "mixed_exact_and_parent"})]
    bands = df[BAND_COLS].to_numpy(dtype=float)
    groups = df["i_atlas_sample"].to_numpy()
    base = _predict_from_bands(bundle, bands)
    base_pm = pd.Series(base, index=groups).groupby(level=0).mean()

    add_shifts, mult_shifts = [], []
    for _ in range(reps):
        b_add = np.clip(bands + rng.normal(0, add_sigma, bands.shape), 1e-4, 0.999)
        b_mul = np.clip(bands * (1 + rng.normal(0, mult_sigma, bands.shape)), 1e-4, 0.999)
        pa = pd.Series(_predict_from_bands(bundle, b_add), index=groups).groupby(level=0).mean()
        pm = pd.Series(_predict_from_bands(bundle, b_mul), index=groups).groupby(level=0).mean()
        add_shifts.append(float((pa - base_pm).abs().mean()))
        mult_shifts.append(float((pm - base_pm).abs().mean()))
    return {
        "additive_sigma": add_sigma,
        "multiplicative_sigma": mult_sigma,
        "mean_parent_pred_shift_additive": float(np.mean(add_shifts)),
        "mean_parent_pred_shift_multiplicative": float(np.mean(mult_shifts)),
    }


def analyze_scenario(base: Path, scenario: str, dataset: str, target: str,
                     best_label: str) -> dict:
    scen_dir = base / scenario
    oof_path = scen_dir / best_label / "oof_predictions.csv"
    report: dict = {"scenario": scenario, "best_model": best_label}
    if not oof_path.exists():
        report["error"] = "no oof predictions"
        return report
    oof = pd.read_csv(oof_path).dropna(subset=["oof_prediction"])

    # source-quality breakdown (parent level)
    by_scope = {}
    for scope, g in oof.groupby("scope"):
        pm = g.groupby("apollo_parent")[["target", "oof_prediction"]].mean()
        err = (pm["oof_prediction"] - pm["target"]).abs()
        by_scope[str(scope)] = {"n_parents": int(pm.shape[0]), "mae_parent": float(err.mean())}
    report["source_quality_breakdown"] = by_scope

    # worst predicted parents
    pm = oof.groupby("apollo_parent")[["target", "oof_prediction"]].mean()
    pm["abs_error"] = (pm["oof_prediction"] - pm["target"]).abs()
    worst = pm.sort_values("abs_error", ascending=False).head(8).reset_index()
    report["worst_parents"] = worst.round(3).to_dict(orient="records")

    # out-of-range predictions
    neg = int((oof["oof_prediction"] < 0).sum())
    report["negative_predictions"] = {"count": neg, "fraction": float(neg / len(oof))}
    if "is_feo" in target.lower():
        tmin, tmax = float(oof["target"].min()), float(oof["target"].max())
        oor = int(((oof["oof_prediction"] < tmin) | (oof["oof_prediction"] > tmax)).sum())
        report["isfeo_out_of_observed_range"] = {
            "observed_min": tmin, "observed_max": tmax,
            "count_outside": oor, "fraction": float(oor / len(oof)),
            "pred_min": float(oof["oof_prediction"].min()),
            "pred_max": float(oof["oof_prediction"].max()),
        }

    # within-parent prediction spread (replicate stability)
    spread = oof.groupby("apollo_parent")["oof_prediction"].std().dropna()
    report["within_parent_pred_spread"] = {
        "mean_std": float(spread.mean()), "max_std": float(spread.max())}

    # perturbation sensitivity via saved model
    model_path = scen_dir / best_label / "final_model.joblib"
    if model_path.exists():
        try:
            bundle = joblib.load(model_path)
            report["perturbation_sensitivity"] = perturbation_sensitivity(
                bundle, dataset, target)
        except Exception as exc:
            report["perturbation_sensitivity"] = {"error": f"{type(exc).__name__}: {exc}"}
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-dir", default="data/ml_experiments_v4")
    p.add_argument("--dataset", required=True)
    a = p.parse_args()
    base = Path(a.base_dir)

    comp = pd.read_csv(base / "comparison_all.csv")
    targets = {"feo_combined": "feo_wt_percent_mean",
               "feo_direct": "feo_direct_wt_percent_mean",
               "isfeo": "is_feo_mean"}
    summary_rows = []
    for scenario, target in targets.items():
        sub = comp[(comp["scenario"] == scenario) & (comp["kind"] == "ml_model")
                   & (comp["status"] == "ok") & (comp["label"] != "null_mean")]
        if sub.empty:
            continue
        best = sub.sort_values("mae_parent").iloc[0]["label"]
        rep = analyze_scenario(base, scenario, a.dataset, target, best)
        (base / f"robustness_{scenario}.json").write_text(json.dumps(rep, indent=2, default=str))
        print(f"\n=== robustness: {scenario} (best={best}) ===")
        print(json.dumps(rep, indent=2, default=str)[:1500])
        ps = rep.get("perturbation_sensitivity", {})
        summary_rows.append({
            "scenario": scenario, "best_model": best,
            "negative_pred_count": rep.get("negative_predictions", {}).get("count"),
            "within_parent_mean_std": rep.get("within_parent_pred_spread", {}).get("mean_std"),
            "pred_shift_add_0.005": ps.get("mean_parent_pred_shift_additive"),
            "pred_shift_mult_2pct": ps.get("mean_parent_pred_shift_multiplicative"),
        })
    pd.DataFrame(summary_rows).to_csv(base / "robustness_summary.csv", index=False)
    print(f"\nwrote {base / 'robustness_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
