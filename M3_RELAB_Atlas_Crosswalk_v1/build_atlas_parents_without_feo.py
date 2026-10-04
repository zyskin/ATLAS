#!/usr/bin/env python3
"""List Atlas parents having RELAB spectra but no known FeO anywhere below them.

Input is the spectrum-level table produced by build_atlas_feo_table.py.  A
parent is omitted as soon as any row belonging to that parent has a known FeO
value, whether the value belongs to the parent itself or to any child.

The result contains one row per remaining parent:
    id, apollo_parent_id, relab_measurement_ids

This program does not claim that FeO is absent from the Atlas PDF.  It only
states that no FeO association was found in the extracted RELAB chemistry.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


OUTPUT_FIELDS = ["id", "apollo_parent_id", "relab_measurement_ids"]


def clean(value: object) -> str:
    return " ".join(str(value or "").split())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a parent-level CSV for Atlas samples represented by RELAB "
            "spectra but having no known FeO for the parent or any child."
        )
    )
    parser.add_argument("--atlas-feo-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--unknown-feo",
        type=float,
        default=-99.0,
        help="Unknown-FeO sentinel used in the input table (default: -99).",
    )
    return parser.parse_args(argv)


def build_rows(input_path: Path, unknown_feo: float) -> tuple[list[dict[str, object]], dict[str, int]]:
    relab_ids: dict[int, set[str]] = defaultdict(set)
    has_known_feo: dict[int, bool] = defaultdict(bool)
    input_rows = 0

    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {
            "i_atlas_sample",
            "k_feo_wt_percent",
            "feo_known",
            "relab_measurement_id",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{input_path} is missing required columns: {', '.join(sorted(missing))}"
            )

        for row in reader:
            input_rows += 1
            parent_text = clean(row["i_atlas_sample"])
            try:
                parent = int(parent_text)
            except ValueError as exc:
                raise ValueError(f"Invalid Atlas parent ID {parent_text!r}") from exc

            # -99 represents unmatched/non-Atlas material and is outside this list.
            if parent < 0:
                continue

            measurement_id = clean(row["relab_measurement_id"])
            if measurement_id:
                relab_ids[parent].add(measurement_id)

            known_flag = clean(row["feo_known"]).casefold()
            try:
                feo = float(clean(row["k_feo_wt_percent"]))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid FeO value {row['k_feo_wt_percent']!r} for parent {parent}"
                ) from exc

            known_by_value = feo != unknown_feo
            known_by_flag = known_flag in {"yes", "true", "1"}
            if known_by_value != known_by_flag:
                raise ValueError(
                    f"Inconsistent FeO flag/value for parent {parent}, "
                    f"RELAB ID {measurement_id}: flag={row['feo_known']!r}, value={feo}"
                )
            has_known_feo[parent] = has_known_feo[parent] or known_by_flag

    output_rows: list[dict[str, object]] = []
    for parent in sorted(relab_ids):
        if has_known_feo[parent]:
            continue
        output_rows.append(
            {
                "id": len(output_rows) + 1,
                "apollo_parent_id": parent,
                "relab_measurement_ids": ",".join(sorted(relab_ids[parent])),
            }
        )

    stats = {
        "input_rows": input_rows,
        "atlas_parents_with_relab": len(relab_ids),
        "parents_with_any_feo": sum(has_known_feo[parent] for parent in relab_ids),
        "parents_without_feo": len(output_rows),
        "listed_relab_spectra": sum(
            len(relab_ids[int(row["apollo_parent_id"])]) for row in output_rows
        ),
    }
    return output_rows, stats


def write_rows(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.atlas_feo_table.is_file():
        print(f"ERROR: input file not found: {args.atlas_feo_table}", file=sys.stderr)
        return 2
    try:
        rows, stats = build_rows(args.atlas_feo_table, args.unknown_feo)
        write_rows(args.output, rows)
    except (OSError, ValueError, csv.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Output: {args.output}")
    print(f"Atlas parents represented by RELAB spectra: {stats['atlas_parents_with_relab']}")
    print(f"Parents with at least one known FeO association: {stats['parents_with_any_feo']}")
    print(f"Parents with no known FeO association: {stats['parents_without_feo']}")
    print(f"RELAB spectrum IDs listed for PDF review: {stats['listed_relab_spectra']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
