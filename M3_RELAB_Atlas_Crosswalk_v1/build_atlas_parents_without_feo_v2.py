#!/usr/bin/env python3
"""Combine the RELAB no-FeO parent list with Atlas-PDF FeO extraction.

This is version 2 of build_atlas_parents_without_feo.py.  The original program
is intentionally retained unchanged.

The baseline population is identical to version 1: Atlas parents represented
by one or more RELAB spectra, with no safely associated RELAB FeO value for the
parent or any child.  Version 2 adds FeO candidates recovered from Atlas PDFs
and writes two outputs:

1. atlas_parents_without_relab_feo_with_pdf_feo_v2.csv
   Every baseline parent, with accepted PDF bulk FeO, review-only PDF
   candidates, and mineral-phase FeO shown in separate columns.

2. atlas_parents_without_feo_in_relab_or_atlas_pdfs.csv
   Only parents having neither RELAB FeO nor any parent-level PDF FeO candidate.

Mineral-phase FeO is never treated as whole-sample/parent FeO.  Ambiguous PDF
values remain explicitly marked for review and are never silently accepted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


VERSION = "atlas-parents-without-feo-v2-1.0"

OUTPUT_ALL_NAME = "atlas_parents_without_relab_feo_with_pdf_feo_v2.csv"
OUTPUT_NONE_NAME = "atlas_parents_without_feo_in_relab_or_atlas_pdfs.csv"
OUTPUT_SUMMARY_NAME = "atlas_parents_without_feo_v2_summary.json"

ALL_FIELDS = [
    "id",
    "apollo_parent_id",
    "relab_measurement_ids",
    "pdf_parent_feo_status",
    "pdf_feo_all_parent_candidates_wt_percent",
    "pdf_feo_accepted_bulk_wt_percent",
    "pdf_feo_review_bulk_wt_percent",
    "pdf_feo_review_unspecified_scope_wt_percent",
    "pdf_feo_mineral_phase_not_parent_wt_percent",
    "pdf_accepted_measurement_ids",
    "pdf_review_measurement_ids",
    "pdf_document_pages",
    "pdf_parent_candidate_count",
    "pdf_mineral_phase_count",
]

NONE_FIELDS = [
    "id",
    "v2_id",
    "apollo_parent_id",
    "relab_measurement_ids",
    "status",
    "pdf_mineral_phase_not_parent_wt_percent",
]


def clean(value: object) -> str:
    return " ".join(str(value or "").split())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--atlas-feo-table",
        type=Path,
        required=True,
        help="RELAB/Atlas table produced by build_atlas_feo_table.py",
    )
    parser.add_argument(
        "--pdf-feo-table",
        type=Path,
        required=True,
        help="atlas_feo_measurements.csv produced by extract_atlas_stage2.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory receiving the two CSV files and summary JSON",
    )
    parser.add_argument(
        "--unknown-feo",
        type=float,
        default=-99.0,
        help="Unknown RELAB FeO sentinel (default: -99)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No CSV header found in {path}")
        return [dict(row) for row in reader], list(reader.fieldnames)


def require_columns(path: Path, fieldnames: list[str], required: Iterable[str]) -> None:
    missing = sorted(set(required).difference(fieldnames))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def parse_parent(value: object, source: str) -> int:
    text = clean(value)
    try:
        parent = int(text)
    except ValueError as exc:
        raise ValueError(f"Invalid Apollo parent ID {text!r} in {source}") from exc
    return parent


def parse_float(value: object, source: str) -> float:
    text = clean(value)
    try:
        result = float(text)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric value {text!r} in {source}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Non-finite numeric value {text!r} in {source}")
    return result


def baseline_without_relab_feo(
    rows: list[dict[str, str]], unknown_feo: float
) -> tuple[dict[int, set[str]], dict[str, int]]:
    relab_ids: dict[int, set[str]] = defaultdict(set)
    has_known: dict[int, bool] = defaultdict(bool)
    for row_number, row in enumerate(rows, 2):
        parent = parse_parent(row.get("i_atlas_sample", ""), f"RELAB row {row_number}")
        if parent < 0:
            continue
        measurement_id = clean(row.get("relab_measurement_id", ""))
        if measurement_id:
            relab_ids[parent].add(measurement_id)
        feo = parse_float(row.get("k_feo_wt_percent", ""), f"RELAB row {row_number}")
        known_by_value = feo != unknown_feo
        known_by_flag = clean(row.get("feo_known", "")).casefold() in {"yes", "true", "1"}
        if known_by_value != known_by_flag:
            raise ValueError(
                f"Inconsistent RELAB FeO flag/value for parent {parent}, "
                f"measurement {measurement_id}: {feo!r} vs {row.get('feo_known')!r}"
            )
        has_known[parent] = has_known[parent] or known_by_flag

    baseline = {parent: ids for parent, ids in relab_ids.items() if not has_known[parent]}
    stats = {
        "atlas_parents_with_relab": len(relab_ids),
        "parents_with_any_relab_feo": sum(has_known[parent] for parent in relab_ids),
        "baseline_parents_without_relab_feo": len(baseline),
    }
    return baseline, stats


def value_text(row: dict[str, str]) -> str:
    text = clean(row.get("value_text", ""))
    if text:
        return text
    return clean(row.get("value_wt_percent", ""))


def value_sort_key(value: str) -> tuple[float, str]:
    numeric = value.lstrip("<>~≈")
    try:
        return float(numeric), value
    except ValueError:
        return math.inf, value


def unique_values(rows: list[dict[str, str]]) -> str:
    values = {value_text(row) for row in rows if value_text(row)}
    return ",".join(sorted(values, key=value_sort_key))


def unique_measurement_ids(rows: list[dict[str, str]]) -> str:
    values = {clean(row.get("measurement_id", "")) for row in rows}
    return ",".join(sorted(value for value in values if value))


def unique_document_pages(rows: list[dict[str, str]]) -> str:
    values = set()
    for row in rows:
        document_id = clean(row.get("document_id", ""))
        page = clean(row.get("pdf_page", ""))
        if document_id or page:
            values.add(f"{document_id}:p{page}")
    return ",".join(sorted(values))


def classify_pdf_rows(
    pdf_rows: list[dict[str, str]], baseline: dict[int, set[str]]
) -> dict[int, dict[str, list[dict[str, str]]]]:
    grouped: dict[int, dict[str, list[dict[str, str]]]] = {
        parent: {
            "accepted_bulk": [],
            "review_bulk": [],
            "review_unspecified": [],
            "mineral_phase": [],
        }
        for parent in baseline
    }

    for row_number, row in enumerate(pdf_rows, 2):
        parent = parse_parent(row.get("atlas_sample_id", ""), f"PDF FeO row {row_number}")
        if parent not in grouped:
            continue
        # Validate the numeric value even though the output retains value_text.
        parse_float(row.get("value_wt_percent", ""), f"PDF FeO row {row_number}")
        scope = clean(row.get("sample_scope", "")).casefold()
        needs_review = clean(row.get("needs_review", "")).casefold() in {
            "yes", "true", "1"
        }
        if scope == "bulk_or_whole_sample":
            key = "review_bulk" if needs_review else "accepted_bulk"
        elif scope == "mineral_phase":
            key = "mineral_phase"
        else:
            # Unspecified scope can be a parent value, but it cannot be accepted
            # until the table context is reviewed.
            key = "review_unspecified"
        grouped[parent][key].append(row)
    return grouped


def status_for(groups: dict[str, list[dict[str, str]]]) -> str:
    accepted = bool(groups["accepted_bulk"])
    review = bool(groups["review_bulk"] or groups["review_unspecified"])
    if accepted and review:
        return "accepted_pdf_bulk_feo_plus_review_candidates"
    if accepted:
        return "accepted_pdf_bulk_feo"
    if review:
        return "review_only_pdf_parent_feo_candidates"
    return "no_parent_level_pdf_feo_candidate"


def build_outputs(
    baseline: dict[int, set[str]],
    grouped: dict[int, dict[str, list[dict[str, str]]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    no_feo_rows: list[dict[str, Any]] = []

    for parent in sorted(baseline):
        groups = grouped[parent]
        accepted = groups["accepted_bulk"]
        review_bulk = groups["review_bulk"]
        review_unspecified = groups["review_unspecified"]
        mineral = groups["mineral_phase"]
        parent_candidates = accepted + review_bulk + review_unspecified
        all_pdf_rows = parent_candidates + mineral
        status = status_for(groups)
        v2_id = len(all_rows) + 1
        all_rows.append(
            {
                "id": v2_id,
                "apollo_parent_id": parent,
                "relab_measurement_ids": ",".join(sorted(baseline[parent])),
                "pdf_parent_feo_status": status,
                "pdf_feo_all_parent_candidates_wt_percent": unique_values(parent_candidates),
                "pdf_feo_accepted_bulk_wt_percent": unique_values(accepted),
                "pdf_feo_review_bulk_wt_percent": unique_values(review_bulk),
                "pdf_feo_review_unspecified_scope_wt_percent": unique_values(review_unspecified),
                "pdf_feo_mineral_phase_not_parent_wt_percent": unique_values(mineral),
                "pdf_accepted_measurement_ids": unique_measurement_ids(accepted),
                "pdf_review_measurement_ids": unique_measurement_ids(
                    review_bulk + review_unspecified
                ),
                "pdf_document_pages": unique_document_pages(all_pdf_rows),
                "pdf_parent_candidate_count": len(parent_candidates),
                "pdf_mineral_phase_count": len(mineral),
            }
        )

        if not parent_candidates:
            no_feo_rows.append(
                {
                    "id": len(no_feo_rows) + 1,
                    "v2_id": v2_id,
                    "apollo_parent_id": parent,
                    "relab_measurement_ids": ",".join(sorted(baseline[parent])),
                    "status": "no FeO found in RELAB metadata or as a parent-level Atlas PDF candidate",
                    "pdf_mineral_phase_not_parent_wt_percent": unique_values(mineral),
                }
            )
    return all_rows, no_feo_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    temporary = path.with_name(path.name + ".part")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for path in (args.atlas_feo_table, args.pdf_feo_table):
        if not path.is_file():
            print(f"ERROR: input file not found: {path}", file=sys.stderr)
            return 2

    output_all = args.output_dir / OUTPUT_ALL_NAME
    output_none = args.output_dir / OUTPUT_NONE_NAME
    output_summary = args.output_dir / OUTPUT_SUMMARY_NAME
    existing = [path for path in (output_all, output_none, output_summary) if path.exists()]
    if existing and not args.overwrite:
        print("ERROR: output exists; add --overwrite to replace it:", file=sys.stderr)
        for path in existing:
            print(f"  {path}", file=sys.stderr)
        return 2

    try:
        relab_rows, relab_fields = read_csv(args.atlas_feo_table)
        pdf_rows, pdf_fields = read_csv(args.pdf_feo_table)
        require_columns(
            args.atlas_feo_table,
            relab_fields,
            ["i_atlas_sample", "k_feo_wt_percent", "feo_known", "relab_measurement_id"],
        )
        require_columns(
            args.pdf_feo_table,
            pdf_fields,
            [
                "measurement_id", "document_id", "atlas_sample_id", "pdf_page",
                "value_wt_percent", "sample_scope", "needs_review",
            ],
        )
        baseline, stats = baseline_without_relab_feo(relab_rows, args.unknown_feo)
        grouped = classify_pdf_rows(pdf_rows, baseline)
        all_rows, no_feo_rows = build_outputs(baseline, grouped)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(output_all, all_rows, ALL_FIELDS)
        write_csv(output_none, no_feo_rows, NONE_FIELDS)

        status_counts: dict[str, int] = defaultdict(int)
        for row in all_rows:
            status_counts[str(row["pdf_parent_feo_status"])] += 1
        summary = {
            "schema": "atlas-parents-without-feo-v2-summary-1.0",
            "program": VERSION,
            "inputs": {
                "atlas_feo_table": str(args.atlas_feo_table),
                "pdf_feo_table": str(args.pdf_feo_table),
            },
            "outputs": {
                "all_baseline_parents": str(output_all),
                "no_feo_in_relab_or_atlas_pdfs": str(output_none),
            },
            "counts": {
                **stats,
                "v2_rows": len(all_rows),
                "parents_without_relab_or_parent_level_pdf_feo": len(no_feo_rows),
                "pdf_status_counts": dict(sorted(status_counts.items())),
            },
            "definitions": {
                "accepted_pdf_bulk_feo": (
                    "sample_scope=bulk_or_whole_sample and needs_review=no"
                ),
                "review_pdf_parent_candidate": (
                    "bulk/whole-sample row marked for review, or sample scope unspecified"
                ),
                "mineral_phase": (
                    "reported separately and never counted as parent/whole-sample FeO"
                ),
                "no_feo_in_relab_or_atlas_pdfs": (
                    "no RELAB FeO and no accepted or review-only parent-level PDF FeO candidate"
                ),
            },
        }
        output_summary.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, csv.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Version-2 review table: {output_all}")
    print(f"No FeO in RELAB or Atlas PDFs: {output_none}")
    print(f"Summary: {output_summary}")
    print(f"Baseline parents without RELAB FeO: {len(all_rows)}")
    for name, count in sorted(status_counts.items()):
        print(f"  {name}: {count}")
    print(f"Parents with no parent-level FeO in either source: {len(no_feo_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
