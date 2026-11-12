#!/usr/bin/env python3
"""Plot every valid pixel of an extracted M3 snapshot on the Lucey plane.

Default output is a logarithmic hexagonal density plot.  Every valid pixel is
used, but pixels in the same hexagon are represented by their count.  Use
--render scatter to draw every point individually.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from lucey_common import (
    KUMAR_ORIGIN,
    LUCEY_ORIGIN,
    load_stage1_lucey_coordinates,
    valid_lucey_values,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-dir", required=True, type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--definition", choices=("direct", "kumar_mean", "both"), default="both"
    )
    parser.add_argument("--render", choices=("hexbin", "scatter"), default="hexbin")
    parser.add_argument("--gridsize", type=int, default=260)
    parser.add_argument("--point-size", type=float, default=0.08)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--reflectance-maximum", type=float, default=2.0)
    parser.add_argument("--ratio-maximum", type=float, default=5.0)
    return parser.parse_args()


def draw_one(args: argparse.Namespace, definition: str) -> dict[str, object]:
    r750, r950, bands = load_stage1_lucey_coordinates(
        args.stage1_dir, args.prefix, definition
    )
    x, ratio, valid = valid_lucey_values(
        r750,
        r950,
        reflectance_maximum=args.reflectance_maximum,
        ratio_maximum=args.ratio_maximum,
    )
    if x.size == 0:
        raise ValueError(f"No valid pixels for {definition}")

    fig, ax = plt.subplots(figsize=(9.2, 7.3), constrained_layout=True)
    if args.render == "hexbin":
        artist = ax.hexbin(
            x,
            ratio,
            gridsize=args.gridsize,
            mincnt=1,
            norm=LogNorm(),
            cmap="viridis",
            linewidths=0,
        )
        colorbar = fig.colorbar(artist, ax=ax)
        colorbar.set_label("valid pixel count per hexagon (log scale)")
    else:
        ax.scatter(
            x, ratio, s=args.point_size, alpha=args.alpha, c="navy",
            linewidths=0, rasterized=True,
        )

    for label, origin, color in (
        ("Lucey origin (0.08, 1.19)", LUCEY_ORIGIN, "crimson"),
        ("Kumar origin (0.08, 1.18)", KUMAR_ORIGIN, "darkorange"),
    ):
        ax.scatter(
            [origin[0]], [origin[1]], marker="X", s=125, c=color,
            edgecolors="black", linewidths=0.7, label=label, zorder=5,
        )

    definition_text = (
        "band 9 and band 19"
        if definition == "direct"
        else "means of bands 8–10 and 18–20"
    )
    ax.set_xlabel(r"$R_{750}$")
    ax.set_ylabel(r"$R_{950}/R_{750}$")
    ax.set_title(
        f"{args.prefix}: all valid M3 pixels on the Lucey plane\n"
        f"{definition_text}; {x.size:,} of {r750.size:,} pixels retained"
    )
    ax.grid(alpha=0.2)
    ax.legend(loc="best")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.prefix}_lucey_plane_{definition}_{args.render}.png"
    fig.savefig(output, dpi=220)
    plt.close(fig)

    return {
        "definition": definition,
        "bands_read": list(bands),
        "source_shape": list(r750.shape),
        "total_pixels": int(r750.size),
        "valid_pixels": int(x.size),
        "excluded_pixels": int(r750.size - x.size),
        "reflectance_maximum": args.reflectance_maximum,
        "ratio_maximum": args.ratio_maximum,
        "render": args.render,
        "output": str(output),
        "r750_range_valid": [float(np.min(x)), float(np.max(x))],
        "ratio_range_valid": [float(np.min(ratio)), float(np.max(ratio))],
    }


def main() -> int:
    args = parse_args()
    definitions = (
        ("direct", "kumar_mean") if args.definition == "both" else (args.definition,)
    )
    summaries = [draw_one(args, definition) for definition in definitions]
    manifest = {
        "stage1_dir": str(args.stage1_dir),
        "prefix": args.prefix,
        "coordinate_order": ["R750", "R950/R750"],
        "lucey_origin": list(LUCEY_ORIGIN),
        "kumar_origin": list(KUMAR_ORIGIN),
        "plots": summaries,
        "note": "All finite pixels passing the explicitly recorded physical plotting limits are used.",
    }
    manifest_path = args.output_dir / f"{args.prefix}_lucey_plane_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for summary in summaries:
        print(
            f"{summary['definition']}: {summary['valid_pixels']:,}/"
            f"{summary['total_pixels']:,} pixels -> {summary['output']}"
        )
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
