#!/usr/bin/env python3
"""Make one ground-truth-versus-prediction plot for every FeO method.

The usual input is ``cv_predictions_by_measurement.csv`` from ML v2 or v3.
All columns beginning with ``prediction_`` are detected, including the Lucey
and Kumar formula variants.  Optional Apollo-parent aggregation gives every
parent one plotted point and avoids visually overweighting parents with many
spectra.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-column", default="feo_wt_percent_mean")
    parser.add_argument(
        "--unit", choices=("measurement", "apollo_parent_mean"),
        default="apollo_parent_mean",
    )
    parser.add_argument("--parent-column", default="i_atlas_sample")
    parser.add_argument("--point-size", type=float, default=30.0)
    parser.add_argument("--alpha", type=float, default=0.72)
    parser.add_argument(
        "--include",
        action="append",
        help="Optional regex; repeat to restrict method names",
    )
    return parser.parse_args()


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    return {
        "n": int(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "bias_prediction_minus_truth": float(np.mean(predicted - actual)),
        "r2": float(r2_score(actual, predicted)) if len(actual) >= 2 else float("nan"),
    }


def resolve_predictions(path: Path) -> Path:
    """Return the requested table or report prediction tables actually present."""
    if path.is_file():
        return path
    project_root = next(
        (parent for parent in path.parents
         if parent.name == "M3_RELAB_Atlas_Crosswalk_v1"),
        None,
    )
    candidates: list[Path] = []
    if project_root is not None:
        data = project_root / "data"
        candidates = sorted(data.glob("**/cv_predictions_by_measurement.csv"))
        candidates += sorted(data.glob("**/cv_predictions.csv"))
    details = ""
    if candidates:
        details = "\\nPrediction tables found instead:\\n" + "\\n".join(
            f"  {candidate}" for candidate in candidates
        )
    raise FileNotFoundError(
        f"Prediction table does not exist: {path}{details}\\n"
        "Use one of the listed held-out cross-validation tables; do not use "
        "final-fit training predictions."
    )


def main() -> int:
    args = parse_args()
    args.predictions = resolve_predictions(args.predictions)
    frame = pd.read_csv(args.predictions, low_memory=False)
    if args.target_column not in frame:
        raise ValueError(f"Target column not found: {args.target_column}")
    prediction_columns = [
        column for column in frame.columns if column.startswith("prediction_")
    ]
    if args.include:
        patterns = [re.compile(pattern) for pattern in args.include]
        prediction_columns = [
            column for column in prediction_columns
            if any(pattern.search(column.removeprefix("prediction_")) for pattern in patterns)
        ]
    if not prediction_columns:
        raise ValueError("No prediction_ columns were found after filtering")
    if args.unit == "apollo_parent_mean" and args.parent_column not in frame:
        raise ValueError(f"Parent column not found: {args.parent_column}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    for prediction_column in prediction_columns:
        method = prediction_column.removeprefix("prediction_")
        table = frame[[
            *([args.parent_column] if args.unit == "apollo_parent_mean" else []),
            args.target_column,
            prediction_column,
        ]].copy()
        table[args.target_column] = pd.to_numeric(table[args.target_column], errors="coerce")
        table[prediction_column] = pd.to_numeric(table[prediction_column], errors="coerce")
        table = table.loc[
            np.isfinite(table[args.target_column]) & np.isfinite(table[prediction_column])
        ]
        if args.unit == "apollo_parent_mean":
            table = table.groupby(args.parent_column, as_index=False)[
                [args.target_column, prediction_column]
            ].mean()
        if table.empty:
            continue
        actual = table[args.target_column].to_numpy(float)
        predicted = table[prediction_column].to_numpy(float)
        score = metrics(actual, predicted)
        limits = [
            float(min(np.min(actual), np.min(predicted))),
            float(max(np.max(actual), np.max(predicted))),
        ]
        padding = max(0.5, 0.04 * (limits[1] - limits[0]))
        limits = [limits[0] - padding, limits[1] + padding]

        fig, ax = plt.subplots(figsize=(6.8, 6.4), constrained_layout=True)
        ax.scatter(
            actual, predicted, s=args.point_size, alpha=args.alpha,
            c="tab:blue", edgecolors="none",
        )
        ax.plot(limits, limits, color="black", linestyle="--", linewidth=1.2)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Ground-truth FeO (wt.%)")
        ax.set_ylabel("Held-out prediction (wt.%)")
        ax.set_title(
            f"{method} — {args.unit}\n"
            f"n={score['n']}; MAE={score['mae']:.3f}; RMSE={score['rmse']:.3f}; "
            f"R²={score['r2']:.3f}"
        )
        ax.grid(alpha=0.22)
        output = args.output_dir / f"feo_truth_vs_prediction_{safe_name(method)}_{args.unit}.png"
        fig.savefig(output, dpi=210)
        plt.close(fig)
        results.append({
            "method": method,
            "evaluation_unit": args.unit,
            **score,
            "plot": str(output),
        })

    summary = pd.DataFrame(results).sort_values("mae", na_position="last")
    summary_csv = args.output_dir / f"feo_truth_vs_prediction_metrics_{args.unit}.csv"
    summary.to_csv(summary_csv, index=False)
    manifest = {
        "input": str(args.predictions),
        "target_column": args.target_column,
        "evaluation_unit": args.unit,
        "method_count": int(len(summary)),
        "summary_csv": str(summary_csv),
        "note": "Predictions must be held-out cross-validation predictions; fitted training predictions are not an honest performance plot.",
    }
    (args.output_dir / f"feo_truth_vs_prediction_manifest_{args.unit}.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(summary[["method", "n", "mae", "rmse", "r2"]].to_string(index=False))
    print(f"Plots: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
