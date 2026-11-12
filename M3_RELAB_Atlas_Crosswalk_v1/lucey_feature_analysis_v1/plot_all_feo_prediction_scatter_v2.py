#!/usr/bin/env python3
"""Plot held-out FeO truth versus prediction across v2, Claude v4, and v5.

The three experiment generations use different storage layouts:

* v2: one wide CSV with many ``prediction_*`` columns;
* Claude v4: one ``oof_predictions.csv`` per model, with columns ``target``
  and ``oof_prediction``;
* bounded v5: one ``oof_predictions.csv`` per model and link function, with
  columns ``target_wt_percent`` and ``prediction_wt_percent``.

Every source must contain held-out, parent-grouped cross-validation
predictions.  Final-fit training predictions are deliberately unsupported.
One PNG is written per method, plus a combined metrics CSV and manifest.
"""

from __future__ import annotations

import argparse
import json
import math
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
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--v2-wide", type=Path)
    parser.add_argument("--v2-target-column", default="feo_wt_percent_mean")
    parser.add_argument("--v2-parent-column", default="i_atlas_sample")
    parser.add_argument(
        "--v4-root", type=Path,
        help="Usually data/ml_experiments_v4/feo_combined",
    )
    parser.add_argument(
        "--v5-root", type=Path,
        help="Usually data/ml_bounded_v5/feo_wt_percent_mean__all",
    )
    parser.add_argument(
        "--unit", choices=("measurement", "apollo_parent_mean"),
        default="apollo_parent_mean",
    )
    parser.add_argument(
        "--include", action="append",
        help="Optional regex on the full method label; repeat if needed",
    )
    parser.add_argument("--point-size", type=float, default=30.0)
    parser.add_argument("--alpha", type=float, default=0.72)
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")


def require_dir(path: Path, description: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{description} does not exist: {path}")


def normalise(
    frame: pd.DataFrame,
    *,
    method: str,
    parent_column: str,
    target_column: str,
    prediction_column: str,
    source_file: Path,
) -> pd.DataFrame:
    required = {parent_column, target_column, prediction_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source_file} lacks columns: {', '.join(missing)}")
    result = frame[[parent_column, target_column, prediction_column]].copy()
    result.columns = ["apollo_parent", "target", "prediction"]
    result["target"] = pd.to_numeric(result["target"], errors="coerce")
    result["prediction"] = pd.to_numeric(result["prediction"], errors="coerce")
    result = result.loc[
        np.isfinite(result["target"]) & np.isfinite(result["prediction"])
    ].copy()
    result["method"] = method
    result["source_file"] = str(source_file)
    return result


def read_v2(path: Path, target: str, parent: str) -> list[pd.DataFrame]:
    require_file(path, "v2 wide prediction table")
    frame = pd.read_csv(path, low_memory=False)
    prediction_columns = [c for c in frame if c.startswith("prediction_")]
    if not prediction_columns:
        raise ValueError(f"No prediction_* columns in {path}")
    return [
        normalise(
            frame,
            method=f"v2__{column.removeprefix('prediction_')}",
            parent_column=parent,
            target_column=target,
            prediction_column=column,
            source_file=path,
        )
        for column in prediction_columns
    ]


def method_from_relative(path: Path, root: Path, prefix: str) -> str:
    relative = path.parent.relative_to(root)
    return prefix + "__" + "__".join(relative.parts)


def read_v4(root: Path) -> list[pd.DataFrame]:
    require_dir(root, "Claude v4 FeO result directory")
    files = sorted(root.glob("*/oof_predictions.csv"))
    if not files:
        raise FileNotFoundError(f"No */oof_predictions.csv files under {root}")
    output = []
    for path in files:
        frame = pd.read_csv(path, low_memory=False)
        output.append(normalise(
            frame,
            method=method_from_relative(path, root, "v4"),
            parent_column="apollo_parent",
            target_column="target",
            prediction_column="oof_prediction",
            source_file=path,
        ))
    return output


def read_v5(root: Path) -> list[pd.DataFrame]:
    require_dir(root, "bounded v5 FeO result directory")
    files = sorted(root.glob("*/*/oof_predictions.csv"))
    if not files:
        raise FileNotFoundError(f"No */*/oof_predictions.csv files under {root}")
    output = []
    for path in files:
        frame = pd.read_csv(path, low_memory=False)
        output.append(normalise(
            frame,
            method=method_from_relative(path, root, "v5"),
            parent_column="apollo_parent",
            target_column="target_wt_percent",
            prediction_column="prediction_wt_percent",
            source_file=path,
        ))
    return output


def score(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    return {
        "n": int(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "bias_prediction_minus_truth": float(np.mean(predicted - actual)),
        "r2": float(r2_score(actual, predicted)) if len(actual) >= 2 else math.nan,
    }


def main() -> int:
    args = parse_args()
    tables: list[pd.DataFrame] = []
    if args.v2_wide:
        tables.extend(read_v2(args.v2_wide, args.v2_target_column, args.v2_parent_column))
    if args.v4_root:
        tables.extend(read_v4(args.v4_root))
    if args.v5_root:
        tables.extend(read_v5(args.v5_root))
    if not tables:
        raise ValueError("Supply at least one of --v2-wide, --v4-root, or --v5-root")

    patterns = [re.compile(p) for p in args.include] if args.include else []
    if patterns:
        tables = [
            table for table in tables
            if any(p.search(str(table["method"].iloc[0])) for p in patterns)
        ]
    if not tables:
        raise ValueError("No methods remain after --include filtering")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    seen: set[str] = set()
    for table in tables:
        method = str(table["method"].iloc[0])
        if method in seen:
            raise ValueError(f"Duplicate method label: {method}")
        seen.add(method)
        source_file = str(table["source_file"].iloc[0])
        if args.unit == "apollo_parent_mean":
            plotted = table.groupby("apollo_parent", as_index=False)[
                ["target", "prediction"]
            ].mean()
        else:
            plotted = table[["target", "prediction"]]
        actual = plotted["target"].to_numpy(float)
        predicted = plotted["prediction"].to_numpy(float)
        metrics = score(actual, predicted)
        low = float(min(actual.min(), predicted.min()))
        high = float(max(actual.max(), predicted.max()))
        padding = max(0.5, 0.04 * (high - low))
        limits = [low - padding, high + padding]

        fig, axis = plt.subplots(figsize=(6.8, 6.4), constrained_layout=True)
        axis.scatter(
            actual, predicted, s=args.point_size, alpha=args.alpha,
            color="tab:blue", edgecolors="none",
        )
        axis.plot(limits, limits, "k--", linewidth=1.2)
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Ground-truth FeO (wt.%)")
        axis.set_ylabel("Held-out prediction (wt.%)")
        axis.set_title(
            f"{method} — {args.unit}\n"
            f"n={metrics['n']}; MAE={metrics['mae']:.3f}; "
            f"RMSE={metrics['rmse']:.3f}; R²={metrics['r2']:.3f}"
        )
        axis.grid(alpha=0.22)
        output = args.output_dir / (
            f"feo_truth_vs_prediction_{safe_name(method)}_{args.unit}.png"
        )
        fig.savefig(output, dpi=210)
        plt.close(fig)
        results.append({
            "method": method,
            "evaluation_unit": args.unit,
            **metrics,
            "source_file": source_file,
            "plot": str(output),
        })

    summary = pd.DataFrame(results).sort_values(["mae", "method"])
    summary_path = args.output_dir / f"all_models_metrics_{args.unit}.csv"
    summary.to_csv(summary_path, index=False)
    manifest = {
        "schema": "feo-all-model-scatter-v2",
        "evaluation_unit": args.unit,
        "method_count": int(len(summary)),
        "sources": {
            "v2_wide": str(args.v2_wide) if args.v2_wide else None,
            "v4_root": str(args.v4_root) if args.v4_root else None,
            "v5_root": str(args.v5_root) if args.v5_root else None,
        },
        "metrics_csv": str(summary_path),
        "warning": (
            "Compare MAE/RMSE only when n and target scope match. All inputs "
            "must be held-out parent-grouped cross-validation predictions."
        ),
    }
    manifest_path = args.output_dir / "all_models_scatter_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(summary[["method", "n", "mae", "rmse", "r2"]].to_string(index=False))
    print(f"Methods plotted: {len(summary)}")
    print(f"Plots: {args.output_dir}")
    print(f"Metrics: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
