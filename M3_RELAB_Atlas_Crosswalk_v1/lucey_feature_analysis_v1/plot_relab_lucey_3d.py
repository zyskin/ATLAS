#!/usr/bin/env python3
"""Plot RELAB spectra in (R750, R950/R750, FeO) coordinates.

The input is the combined ground-truth table.  Each plotted point is one RELAB
measurement with a finite FeO association.  This is deliberately a spectrum-
level plot: several measurements may share an Apollo parent and the same FeO.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lucey_common import KUMAR_ORIGIN, LUCEY_ORIGIN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--output-png", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument(
        "--target",
        choices=("combined", "direct"),
        default="combined",
        help="combined uses all accepted FeO sources; direct requires specimen-level FeO",
    )
    parser.add_argument("--elevation", type=float, default=24.0)
    parser.add_argument("--azimuth", type=float, default=-58.0)
    parser.add_argument("--point-size", type=float, default=18.0)
    parser.add_argument("--alpha", type=float, default=0.72)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = pd.read_csv(args.ground_truth, low_memory=False)
    target = (
        "feo_wt_percent_mean"
        if args.target == "combined"
        else "feo_direct_wt_percent_mean"
    )
    required = {"r750_direct_b09", "r950_over_r750_direct", target}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    selected = frame.copy()
    for column in required:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    selected = selected.loc[
        np.isfinite(selected["r750_direct_b09"])
        & np.isfinite(selected["r950_over_r750_direct"])
        & np.isfinite(selected[target])
        & (selected["r750_direct_b09"] > 0)
        & (selected["r950_over_r750_direct"] > 0)
    ].copy()
    if selected.empty:
        raise ValueError("No rows have finite Lucey coordinates and FeO")

    x = selected["r750_direct_b09"].to_numpy(float)
    y = selected["r950_over_r750_direct"].to_numpy(float)
    z = selected[target].to_numpy(float)

    figure = plt.figure(figsize=(11.5, 8.5), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    points = axis.scatter(
        x,
        y,
        z,
        c=z,
        cmap="viridis",
        s=args.point_size,
        alpha=args.alpha,
        linewidths=0.15,
        edgecolors="black",
    )
    colorbar = figure.colorbar(points, ax=axis, shrink=0.72, pad=0.09)
    colorbar.set_label("Associated FeO (wt.%)")

    z0 = max(0.0, float(np.nanmin(z)) - 0.06 * float(np.ptp(z)))
    z1 = float(np.nanmax(z))
    for label, origin, color in (
        ("Lucey origin", LUCEY_ORIGIN, "crimson"),
        ("Kumar origin", KUMAR_ORIGIN, "darkorange"),
    ):
        axis.plot(
            [origin[0], origin[0]],
            [origin[1], origin[1]],
            [z0, z1],
            linestyle="--",
            color=color,
            linewidth=1.4,
            alpha=0.85,
        )
        axis.scatter(
            [origin[0]], [origin[1]], [z0], marker="X", s=110, c=color,
            edgecolors="black", linewidths=0.6, label=label,
        )

    axis.set_xlabel(r"$R_{750}$ (M3 band 9 equivalent)", labelpad=10)
    axis.set_ylabel(r"$R_{950}/R_{750}$", labelpad=10)
    axis.set_zlabel("FeO (wt.%)", labelpad=8)
    axis.set_title(
        f"RELAB spectra on the Lucey plane; FeO target={args.target}\n"
        f"{len(selected)} spectra from "
        f"{selected.get('i_atlas_sample', pd.Series(dtype=object)).nunique()} Apollo parents"
    )
    axis.view_init(elev=args.elevation, azim=args.azimuth)
    axis.legend(loc="upper left")

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_png, dpi=220)
    plt.close(figure)

    output_csv = args.output_csv or args.output_png.with_suffix(".csv")
    columns = [
        column for column in (
            "id", "i_atlas_sample", "j_child", "relab_measurement_id",
            "relab_specimen_id", "feo_target_scope", "r750_direct_b09",
            "r950_direct_b19", "r950_over_r750_direct", target,
            "feo_wt_percent_min", "feo_wt_percent_max",
        ) if column in selected.columns
    ]
    selected[columns].to_csv(output_csv, index=False)
    summary = {
        "input": str(args.ground_truth),
        "target": target,
        "spectra_plotted": int(len(selected)),
        "apollo_parents": int(selected["i_atlas_sample"].nunique())
        if "i_atlas_sample" in selected else None,
        "lucey_origin_R750_ratio": list(LUCEY_ORIGIN),
        "kumar_origin_R750_ratio": list(KUMAR_ORIGIN),
        "output_png": str(args.output_png),
        "output_csv": str(output_csv),
    }
    args.output_png.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Plotted RELAB spectra: {len(selected)}")
    print(f"Plot: {args.output_png}")
    print(f"Points: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
