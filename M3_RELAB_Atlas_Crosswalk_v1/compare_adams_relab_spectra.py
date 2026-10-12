#!/usr/bin/env python3
"""Compare one Adams spectrum with modern RELAB spectra for one Apollo sample.

The program deliberately distinguishes an exact child/subsample match from a
same-parent-only match.  Curves are plotted on their native wavelength grids;
no interpolation or smoothing is applied.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Plot an Adams wavelength-corrected spectrum against modern RELAB "
            "spectra belonging to the same Apollo parent or exact child."
        )
    )
    p.add_argument("--apollo-parent", required=True, help="Five-digit Apollo sample ID")
    p.add_argument(
        "--child",
        help="Optional child/subsample after the comma, for example 49 or 9002A",
    )
    p.add_argument("--adams-id", required=True, help="Full Adams ID, including suffix")
    p.add_argument(
        "--cls-id",
        help="Corrected Adams measurement ID, for example CLS391; required if Adams ID is ambiguous",
    )
    p.add_argument(
        "--relab-id",
        action="append",
        default=[],
        help="Modern RELAB measurement ID to plot; repeat for several curves",
    )
    p.add_argument(
        "--all-relab",
        action="store_true",
        help="Plot every matching modern RELAB curve (can be visually crowded)",
    )
    p.add_argument(
        "--list-only",
        action="store_true",
        help="List matching modern RELAB records without making a plot",
    )
    p.add_argument(
        "--normalize-nm",
        type=float,
        help="Divide each curve by its linearly interpolated reflectance at this wavelength",
    )
    p.add_argument("--wavelength-min", type=float, default=350.0)
    p.add_argument("--wavelength-max", type=float, default=2600.0)
    p.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="M3_RELAB_Atlas_Crosswalk_v1 directory",
    )
    p.add_argument("--output", type=Path, help="Output PNG path")
    p.add_argument("--no-show", action="store_true", help="Save without opening a plot window")
    return p


def clean(value: str) -> str:
    return value.strip().upper().removesuffix(".DAT").removesuffix(".TXT")


def load_adams(project: Path, adams_id: str, cls_id: str | None):
    base = project / "data" / "adams-20220223T225406Z-001" / "adams"
    catalogue = base / "Adams_Spectra.txt"
    corrected_dir = base / "wavelength_corrected_data"

    wanted_adams = clean(adams_id)
    wanted_cls = clean(cls_id) if cls_id else None
    matches = []

    with catalogue.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            sample_id = clean(row["Sample ID"])
            corrected = clean(row["Wavelength Corrected File"])
            if sample_id == wanted_adams and (wanted_cls is None or corrected == wanted_cls):
                matches.append((sample_id, corrected, row))

    if not matches:
        raise ValueError(f"No Adams catalogue row matches {adams_id!r} and CLS={cls_id!r}")
    if len(matches) > 1:
        choices = ", ".join(sorted({m[1] for m in matches}))
        raise ValueError(
            f"Adams ID {adams_id!r} has several corrected spectra: {choices}. "
            "Select one with --cls-id."
        )

    sample_id, corrected, catalogue_row = matches[0]
    spectrum_file = corrected_dir / f"{corrected.lower()}.txt"
    if not spectrum_file.is_file():
        raise FileNotFoundError(f"Corrected Adams spectrum not found: {spectrum_file}")

    wavelength_micron = []
    reflectance = []
    with spectrum_file.open(encoding="utf-8-sig") as handle:
        next(handle, None)  # descriptive line
        next(handle, None)  # column names
        for line_number, line in enumerate(handle, 3):
            fields = line.split()
            if len(fields) < 2:
                continue
            try:
                wavelength_micron.append(float(fields[0]))
                reflectance.append(float(fields[1]))
            except ValueError as exc:
                raise ValueError(f"Invalid Adams number at {spectrum_file}:{line_number}") from exc

    return {
        "adams_id": sample_id,
        "cls_id": corrected,
        "file": spectrum_file,
        "catalogue_row": catalogue_row,
        "wavelength_nm": np.asarray(wavelength_micron) * 1000.0,
        "reflectance": np.asarray(reflectance),
    }


def relab_matches_parent(specimen_name: str, parent: str, child: str | None) -> bool:
    if child is None:
        return re.match(rf"^\s*{re.escape(parent)}(?:\D|$)", specimen_name) is not None

    # RELAB names normally use parent,child; accept a decimal separator as well.
    return (
        re.match(
            rf"^\s*{re.escape(parent)}\s*[,\.]\s*{re.escape(child)}(?:\s|\W|$)",
            specimen_name,
            flags=re.IGNORECASE,
        )
        is not None
    )


def load_relab_index(project: Path, parent: str, child: str | None):
    index_file = project / "relab_raw_native_2026-10-04" / "relab_spectrum_index.csv"
    with index_file.open(newline="", encoding="utf-8-sig") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if relab_matches_parent(row["specimen_name"], parent, child)
        ]
    rows.sort(key=lambda row: row["measurement_id"])
    return rows, index_file


def print_candidates(rows, parent: str, child: str | None) -> None:
    level = f"exact child {parent},{child}" if child else f"parent {parent}"
    print(f"Modern RELAB candidates for {level}: {len(rows)}")
    for row in rows:
        print(
            f"  {row['measurement_id']:<14} {row['specimen_id']:<16} "
            f"{row['specimen_name']}"
        )


def load_relab_points(project: Path, measurement_ids: set[str]):
    points_file = project / "relab_raw_native_2026-10-04" / "relab_spectral_points.csv"
    points = {measurement_id: ([], []) for measurement_id in measurement_ids}
    with points_file.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            measurement_id = row["measurement_id"]
            if measurement_id not in points:
                continue
            try:
                wavelength = float(row["wavelength_nm"])
                reflectance = float(row["reflectance"])
            except ValueError:
                continue
            if np.isfinite(wavelength) and np.isfinite(reflectance):
                points[measurement_id][0].append(wavelength)
                points[measurement_id][1].append(reflectance)

    return {
        measurement_id: (np.asarray(values[0]), np.asarray(values[1]))
        for measurement_id, values in points.items()
    }, points_file


def normalized_at(wavelength, reflectance, wavelength_nm: float | None):
    if wavelength_nm is None:
        return reflectance, None
    order = np.argsort(wavelength)
    x = wavelength[order]
    y = reflectance[order]
    if wavelength_nm < x[0] or wavelength_nm > x[-1]:
        raise ValueError(
            f"Cannot normalize at {wavelength_nm:g} nm: curve covers {x[0]:g}-{x[-1]:g} nm"
        )
    reference = float(np.interp(wavelength_nm, x, y))
    if not np.isfinite(reference) or reference == 0:
        raise ValueError(f"Invalid normalization value {reference} at {wavelength_nm:g} nm")
    return reflectance / reference, reference


def main() -> int:
    args = parser().parse_args()
    project = args.project_dir.resolve()
    parent = args.apollo_parent.strip()
    child = args.child.strip() if args.child else None
    if re.fullmatch(r"\d{5}", parent) is None:
        raise ValueError("--apollo-parent must contain exactly five digits")

    adams = load_adams(project, args.adams_id, args.cls_id)
    candidates, index_file = load_relab_index(project, parent, child)
    print_candidates(candidates, parent, child)

    if args.list_only:
        return 0
    if not candidates:
        level = f"{parent},{child}" if child else parent
        raise ValueError(f"No modern RELAB spectrum in the extracted index matches {level}")

    candidate_by_id = {row["measurement_id"]: row for row in candidates}
    if args.relab_id:
        unknown = [item for item in args.relab_id if item not in candidate_by_id]
        if unknown:
            raise ValueError(
                "Requested RELAB IDs are not matches: " + ", ".join(unknown)
            )
        selected = [candidate_by_id[item] for item in args.relab_id]
    elif args.all_relab:
        selected = candidates
    elif len(candidates) == 1:
        selected = candidates
    else:
        raise ValueError(
            "Several modern RELAB spectra match. Re-run with one or more "
            "--relab-id values, or use --all-relab."
        )

    selected_ids = {row["measurement_id"] for row in selected}
    relab_points, points_file = load_relab_points(project, selected_ids)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    adams_y, adams_reference = normalized_at(
        adams["wavelength_nm"], adams["reflectance"], args.normalize_nm
    )
    ax.plot(
        adams["wavelength_nm"],
        adams_y,
        color="black",
        linewidth=2.4,
        label=f"Adams {adams['adams_id']} / {adams['cls_id']}",
        zorder=4,
    )

    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(selected))))
    for color, row in zip(colors, selected):
        measurement_id = row["measurement_id"]
        wavelength, reflectance = relab_points[measurement_id]
        if wavelength.size == 0:
            print(f"WARNING: no extracted points for {measurement_id}", file=sys.stderr)
            continue
        plotted, reference = normalized_at(wavelength, reflectance, args.normalize_nm)
        ax.plot(
            wavelength,
            plotted,
            color=color,
            linewidth=1.5,
            alpha=0.90,
            label=f"RELAB {measurement_id}: {row['specimen_name']}",
        )
        if reference is not None:
            print(f"Normalization {measurement_id}: {reference:.9g} at {args.normalize_nm:g} nm")

    if adams_reference is not None:
        print(
            f"Normalization {adams['cls_id']}: {adams_reference:.9g} "
            f"at {args.normalize_nm:g} nm"
        )

    comparison_level = "exact child/subsample" if child else "same parent only"
    sample_label = f"{parent},{child}" if child else parent
    ax.set_title(f"Adams versus modern RELAB: Apollo {sample_label} ({comparison_level})")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(
        f"Reflectance normalized at {args.normalize_nm:g} nm"
        if args.normalize_nm is not None
        else "Reflectance"
    )
    ax.set_xlim(args.wavelength_min, args.wavelength_max)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()

    output = args.output
    if output is None:
        suffix = f"_{child}" if child else ""
        output = project / "adams_relab_comparisons" / (
            f"Apollo_{parent}{suffix}_{adams['cls_id']}_comparison.png"
        )
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)

    print(f"Comparison level: {comparison_level}")
    print(f"Adams source: {adams['file']}")
    print(f"RELAB index: {index_file}")
    print(f"RELAB points: {points_file}")
    print(f"Plot: {output}")
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
