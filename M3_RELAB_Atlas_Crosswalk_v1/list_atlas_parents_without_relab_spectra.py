#!/usr/bin/env python3
"""List Lunar Sample Atlas parents with no downloaded RELAB spectrum.

The RELAB crosswalk has already reduced names such as ``72435,77`` to the
Apollo parent ``72435``.  Consequently a parent is counted as covered when
the crosswalk contains a spectrum for the parent itself OR for any child,
fraction, grain-size separate, or other specimen linked to that parent.

No FeO filter and no M3-band-coverage filter is applied here.  This script
answers only: "Do our downloaded/crosswalked RELAB files contain at least one
spectrum belonging to this Atlas parent family?"
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


PARENT_RE = re.compile(r"(?<!\d)(\d{5})(?!\d)")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--atlas-csv",
        required=True,
        type=Path,
        help="atlas_samples.csv",
    )
    parser.add_argument(
        "--relab-crosswalk-csv",
        required=True,
        type=Path,
        help=(
            "CSV connecting RELAB measurements to Atlas parents; normally "
            "atlas_relab_feo_table.csv or relab_atlas_crosswalk.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory in which the result CSVs and summary JSON are written",
    )
    return parser.parse_args()


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No CSV header in {path}")
        return list(reader.fieldnames), list(reader)


def parent_id(value: object) -> str:
    """Return the first five-digit Apollo root ID, or an empty string."""
    match = PARENT_RE.search(str(value or ""))
    return match.group(1) if match else ""


def choose_column(columns: list[str], choices: tuple[str, ...], path: Path) -> str:
    for name in choices:
        if name in columns:
            return name
    raise ValueError(
        f"{path} needs one of these columns: {', '.join(choices)}; "
        f"found: {', '.join(columns)}"
    )


def joined(values: set[str]) -> str:
    return ";".join(sorted(value for value in values if value))


def main() -> None:
    args = arguments()
    atlas_columns, atlas_rows = read_rows(args.atlas_csv)
    relab_columns, relab_rows = read_rows(args.relab_crosswalk_csv)

    if "atlas_sample_id" not in atlas_columns:
        raise ValueError(f"{args.atlas_csv} lacks atlas_sample_id")

    relab_parent_column = choose_column(
        relab_columns,
        ("i_atlas_sample", "apollo_parent_id", "atlas_sample_id"),
        args.relab_crosswalk_csv,
    )
    measurement_column = choose_column(
        relab_columns,
        ("relab_measurement_id", "measurement_id"),
        args.relab_crosswalk_csv,
    )

    # Several Atlas entries can share a five-digit root (for example an
    # alphanumeric variant).  Collapse them into one parent-family record while
    # retaining every original Atlas identifier for auditability.
    atlas_by_parent: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    atlas_without_parent: list[dict[str, str]] = []
    atlas_fields = (
        "atlas_sample_id",
        "mission",
        "sample_type",
        "lithology",
        "description",
        "detail_url",
        "atlas_pdf_url",
    )
    for row_number, row in enumerate(atlas_rows, 2):
        parent = parent_id(row.get("atlas_sample_id"))
        if not parent:
            atlas_without_parent.append(
                {
                    "csv_row": str(row_number),
                    "atlas_sample_id": (row.get("atlas_sample_id") or "").strip(),
                    "reason": "no unambiguous five-digit Apollo parent ID",
                }
            )
            continue
        for field in atlas_fields:
            value = (row.get(field) or "").strip()
            if value:
                atlas_by_parent[parent][field].add(value)

    measurements_by_parent: dict[str, set[str]] = defaultdict(set)
    specimens_by_parent: dict[str, set[str]] = defaultdict(set)
    children_by_parent: dict[str, set[str]] = defaultdict(set)
    relab_without_parent: list[dict[str, str]] = []

    specimen_column = (
        "relab_specimen_id" if "relab_specimen_id" in relab_columns
        else "specimen_id" if "specimen_id" in relab_columns
        else ""
    )
    child_column = "j_child" if "j_child" in relab_columns else ""

    for row_number, row in enumerate(relab_rows, 2):
        raw_parent = (row.get(relab_parent_column) or "").strip()
        parent = parent_id(raw_parent)
        measurement = (row.get(measurement_column) or "").strip()
        if not parent:
            relab_without_parent.append(
                {
                    "csv_row": str(row_number),
                    "relab_measurement_id": measurement,
                    "raw_parent_value": raw_parent,
                    "reason": "no unambiguous five-digit Apollo parent ID",
                }
            )
            continue
        if measurement:
            measurements_by_parent[parent].add(measurement)
        if specimen_column:
            specimen = (row.get(specimen_column) or "").strip()
            if specimen:
                specimens_by_parent[parent].add(specimen)
        if child_column:
            child = (row.get(child_column) or "").strip()
            if child:
                children_by_parent[parent].add(child)

    output_fields = [
        "id",
        "apollo_parent_id",
        "atlas_entry_ids",
        "mission",
        "sample_type",
        "lithology",
        "description",
        "atlas_detail_urls",
        "atlas_pdf_urls",
        "relab_measurement_count",
        "coverage_status",
    ]

    uncovered: list[dict[str, object]] = []
    covered: list[dict[str, object]] = []
    for parent in sorted(atlas_by_parent, key=int):
        atlas = atlas_by_parent[parent]
        measurements = measurements_by_parent.get(parent, set())
        base: dict[str, object] = {
            "apollo_parent_id": parent,
            "atlas_entry_ids": joined(atlas["atlas_sample_id"]),
            "mission": joined(atlas["mission"]),
            "sample_type": joined(atlas["sample_type"]),
            "lithology": joined(atlas["lithology"]),
            "description": " | ".join(sorted(atlas["description"])),
            "atlas_detail_urls": joined(atlas["detail_url"]),
            "atlas_pdf_urls": joined(atlas["atlas_pdf_url"]),
            "relab_measurement_count": len(measurements),
        }
        if measurements:
            base["coverage_status"] = "covered_by_parent_or_child_spectrum"
            covered.append(base)
        else:
            base["coverage_status"] = "no_relab_spectrum_for_parent_or_children"
            uncovered.append(base)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    def write_table(path: Path, rows: list[dict[str, object]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=output_fields)
            writer.writeheader()
            for sequence, row in enumerate(rows, 1):
                writer.writerow({"id": sequence, **row})

    uncovered_path = args.output_dir / "atlas_parents_without_relab_spectral_coverage.csv"
    covered_path = args.output_dir / "atlas_parents_with_relab_spectral_coverage.csv"
    write_table(uncovered_path, uncovered)
    write_table(covered_path, covered)

    unmatched_atlas_path = args.output_dir / "atlas_entries_without_five_digit_parent.csv"
    with unmatched_atlas_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("csv_row", "atlas_sample_id", "reason")
        )
        writer.writeheader()
        writer.writerows(atlas_without_parent)

    unmatched_relab_path = args.output_dir / "relab_rows_without_five_digit_parent.csv"
    with unmatched_relab_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "csv_row",
                "relab_measurement_id",
                "raw_parent_value",
                "reason",
            ),
        )
        writer.writeheader()
        writer.writerows(relab_without_parent)

    covered_measurements = {
        measurement
        for parent in atlas_by_parent
        for measurement in measurements_by_parent.get(parent, set())
    }
    summary = {
        "schema": "atlas-relab-parent-spectral-coverage-1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "atlas_csv": str(args.atlas_csv.resolve()),
            "relab_crosswalk_csv": str(args.relab_crosswalk_csv.resolve()),
            "relab_parent_column": relab_parent_column,
            "relab_measurement_column": measurement_column,
        },
        "definition": (
            "An Atlas parent is covered if at least one downloaded RELAB "
            "measurement is crosswalked to the parent itself or any child/fraction."
        ),
        "filters_not_applied": [
            "FeO availability",
            "wavelength range",
            "M3 74-band completeness",
            "instrument or viewing geometry",
        ],
        "counts": {
            "atlas_input_rows": len(atlas_rows),
            "atlas_unique_five_digit_parents": len(atlas_by_parent),
            "atlas_parents_with_relab_coverage": len(covered),
            "atlas_parents_without_relab_coverage": len(uncovered),
            "unique_relab_measurements_covering_atlas_parents": len(covered_measurements),
            "atlas_rows_without_five_digit_parent": len(atlas_without_parent),
            "relab_rows_without_five_digit_parent": len(relab_without_parent),
        },
        "outputs": {
            "uncovered": str(uncovered_path.resolve()),
            "covered": str(covered_path.resolve()),
            "atlas_unmatched": str(unmatched_atlas_path.resolve()),
            "relab_unmatched": str(unmatched_relab_path.resolve()),
        },
    }
    summary_path = args.output_dir / "atlas_relab_spectral_coverage_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Atlas parent families: {len(atlas_by_parent)}")
    print(f"With RELAB spectral coverage: {len(covered)}")
    print(f"Without RELAB spectral coverage: {len(uncovered)}")
    print(f"Uncovered list: {uncovered_path}")
    print(f"Covered audit table: {covered_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
