#!/usr/bin/env python3
"""Build an auditable Apollo Atlas / RELAB / FeO CSV table.

The output contains one row per RELAB spectrum and associated chemistry record.
If a spectrum has several applicable chemistry records, it receives several
rows.  If no FeO value can be associated safely, the spectrum receives one row
with k_feo_wt_percent = -99.

Rows are sorted lexicographically by (i, j, k):
    i = Apollo Atlas parent sample number
    j = child/subsample number (0 means no child number was found)
    k = FeO weight percent (-99 means unknown)

Chemistry association is deliberately conservative.  It uses an explicit
chemistry LID reference in the spectrum label and, by default, an exact RELAB
specimen-ID match.  It never transfers parent chemistry to a child merely
because their Apollo parent number agrees.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


OUTPUT_FIELDS = [
    "id",
    "i_atlas_sample",
    "j_child",
    "k_feo_wt_percent",
    "feo_known",
    "relab_measurement_id",
    "relab_specimen_id",
    "relab_specimen_name",
    "feo_association",
    "chemistry_product_id",
    "chemistry_specimen_id",
    "chemistry_phase",
    "chemistry_source_and_method",
    "atlas_sample_type",
    "atlas_lithology",
    "age_or_treatment_notes",
    "comments",
    "crosswalk_match_type",
    "crosswalk_needs_review",
    "atlas_detail_url",
    "atlas_pdf_url",
]


AGE_TREATMENT_RE = re.compile(
    r"\b(?:age|aged|aging|matur(?:e|ed|ity)|weather(?:ed|ing)?|"
    r"irradiat(?:ed|ion)?|implant(?:ed|ation)?|heat(?:ed|ing)?|"
    r"anneal(?:ed|ing)?|laser|expos(?:ed|ure)|treated|treatment|"
    r"fresh|agglutinate|agglutinitic|\d+(?:\.\d+)?\s*(?:ga|ma))\b",
    re.IGNORECASE,
)


def clean(value: object) -> str:
    """Normalize whitespace without changing substantive text."""
    return " ".join(str(value or "").split())


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No CSV header found in {path}")
        rows = [dict(row) for row in reader]
        return rows, list(reader.fieldnames)


def require_columns(path: Path, columns: Iterable[str], fieldnames: list[str]) -> None:
    missing = [column for column in columns if column not in fieldnames]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def parse_parent(value: str) -> int | None:
    text = clean(value)
    if not text:
        return None
    match = re.search(r"(?<!\d)(\d{5})(?!\d)", text.replace(",", ""))
    return int(match.group(1)) if match else None


def parse_child(parent: int | None, *candidate_texts: str) -> str:
    """Extract a child/subsample token following a five-digit Apollo parent.

    Handles common forms such as 15555,965; 15531.22; and 10,003,173.
    Returns an empty string when no child number is stated.
    """
    if parent is None:
        return ""

    parent_text = str(parent)
    thousands_parent = f"{parent_text[:2]},{parent_text[2:]}"

    for candidate in candidate_texts:
        text = clean(candidate)
        if not text:
            continue

        patterns = [
            rf"(?<!\d){re.escape(parent_text)}\s*[,.;:/-]\s*(\d+[A-Za-z]?)",
            rf"(?<!\d){re.escape(thousands_parent)}\s*[,.;:/-]\s*(\d+[A-Za-z]?)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1)
    return ""


def child_sort_key(value: str) -> tuple[int, int, str]:
    """Put childless parent records first, then numeric/alphanumeric children."""
    text = clean(value)
    if not text:
        return (0, 0, "")
    match = re.fullmatch(r"(\d+)([A-Za-z]*)", text)
    if match:
        return (1, int(match.group(1)), match.group(2).upper())
    return (2, 0, text.upper())


def split_lid_references(value: str) -> list[str]:
    text = clean(value)
    if not text:
        return []
    # Current extractor uses a pipe for multiple values.  Semicolon support is
    # included for defensive compatibility with future exports.
    return [part.strip() for part in re.split(r"[|;]", text) if part.strip()]


def unique_join(label_values: Iterable[tuple[str, str]]) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for label, value in label_values:
        value = clean(value)
        if not value:
            continue
        item = f"{label}: {value}"
        key = item.casefold()
        if key not in seen:
            parts.append(item)
            seen.add(key)
    return "; ".join(parts)


def age_treatment_notes(values: Iterable[str]) -> str:
    selected: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = clean(value)
        if value and AGE_TREATMENT_RE.search(value) and value.casefold() not in seen:
            selected.append(value)
            seen.add(value.casefold())
    return "; ".join(selected)


def feo_value(chemistry_row: dict[str, str] | None, unknown: float) -> float:
    if chemistry_row is None:
        return unknown
    text = clean(chemistry_row.get("feo_percent", ""))
    if not text:
        return unknown
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(
            f"Invalid feo_percent={text!r} in chemistry product "
            f"{chemistry_row.get('chemistry_product_id', '')!r}"
        ) from exc


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, object]], Counter]:
    spectra, spectrum_fields = read_csv(args.spectrum_index)
    crosswalk, crosswalk_fields = read_csv(args.crosswalk)
    chemistry, chemistry_fields = read_csv(args.chemistry)

    require_columns(
        args.spectrum_index,
        ["measurement_id", "specimen_id", "specimen_name", "chemistry_lid_references"],
        spectrum_fields,
    )
    require_columns(
        args.crosswalk,
        ["measurement_id", "apollo_parent_sample", "atlas_sample_id"],
        crosswalk_fields,
    )
    require_columns(
        args.chemistry,
        ["logical_identifier", "specimen_id", "feo_percent"],
        chemistry_fields,
    )

    crosswalk_by_measurement: dict[str, dict[str, str]] = {}
    duplicate_crosswalk_ids: set[str] = set()
    for row in crosswalk:
        measurement_id = clean(row.get("measurement_id", ""))
        if not measurement_id:
            continue
        if measurement_id in crosswalk_by_measurement:
            duplicate_crosswalk_ids.add(measurement_id)
        else:
            crosswalk_by_measurement[measurement_id] = row
    if duplicate_crosswalk_ids:
        sample = ", ".join(sorted(duplicate_crosswalk_ids)[:10])
        raise ValueError(f"Crosswalk contains duplicate measurement_id values: {sample}")

    chemistry_by_lid: dict[str, list[dict[str, str]]] = defaultdict(list)
    chemistry_by_specimen: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in chemistry:
        lid = clean(row.get("logical_identifier", ""))
        specimen_id = clean(row.get("specimen_id", "")).casefold()
        if lid:
            chemistry_by_lid[lid].append(row)
        if specimen_id:
            chemistry_by_specimen[specimen_id].append(row)

    output_rows: list[dict[str, object]] = []
    stats: Counter = Counter()

    for spectrum in spectra:
        measurement_id = clean(spectrum.get("measurement_id", ""))
        if not measurement_id:
            stats["spectra_missing_measurement_id"] += 1
            continue

        cross = crosswalk_by_measurement.get(measurement_id)
        if cross is None:
            stats["spectra_without_crosswalk"] += 1
            if not args.include_unmatched:
                continue
            cross = {}

        # atlas_sample_id is the preferred value.  apollo_parent_sample is
        # retained only for --include-unmatched records absent from the Atlas.
        atlas_parent = parse_parent(cross.get("atlas_sample_id", ""))
        parent_source = "atlas_sample_id"
        if atlas_parent is None and args.include_unmatched:
            atlas_parent = parse_parent(cross.get("apollo_parent_sample", ""))
            parent_source = "apollo_parent_sample_not_in_atlas"

        if atlas_parent is None and not args.include_unmatched:
            stats["spectra_without_atlas_sample"] += 1
            continue

        spectrum_specimen_id = clean(spectrum.get("specimen_id", ""))
        specimen_name = clean(spectrum.get("specimen_name", ""))
        crosswalk_specimen_name = clean(cross.get("relab_specimen_name", ""))
        child = parse_child(atlas_parent, specimen_name, crosswalk_specimen_name)

        associations: list[tuple[str, dict[str, str] | None]] = []
        seen_chemistry_records: set[tuple[str, str]] = set()

        for lid in split_lid_references(spectrum.get("chemistry_lid_references", "")):
            matched = chemistry_by_lid.get(lid, [])
            if not matched:
                stats["explicit_references_not_downloaded"] += 1
            for chemistry_row in matched:
                record_key = (
                    clean(chemistry_row.get("logical_identifier", "")),
                    clean(chemistry_row.get("chemistry_record_index", "")),
                )
                if record_key not in seen_chemistry_records:
                    associations.append(("explicit_reference", chemistry_row))
                    seen_chemistry_records.add(record_key)

        if args.exact_specimen_match and spectrum_specimen_id:
            for chemistry_row in chemistry_by_specimen.get(spectrum_specimen_id.casefold(), []):
                record_key = (
                    clean(chemistry_row.get("logical_identifier", "")),
                    clean(chemistry_row.get("chemistry_record_index", "")),
                )
                if record_key not in seen_chemistry_records:
                    associations.append(("exact_specimen_id", chemistry_row))
                    seen_chemistry_records.add(record_key)

        if not associations:
            associations = [("none", None)]

        for association, chemistry_row in associations:
            feo = feo_value(chemistry_row, args.unknown_feo)
            known = feo != args.unknown_feo

            chemistry_row = chemistry_row or {}
            chemistry_phase = clean(
                chemistry_row.get("phase_name", "")
                or chemistry_row.get("phase_abbreviation", "")
            )

            descriptive_values = [
                specimen_name,
                spectrum.get("specimen_description", ""),
                spectrum.get("specimen_collection_location", ""),
                spectrum.get("specimen_type", ""),
                spectrum.get("material_origin", ""),
                spectrum.get("material_type", ""),
                spectrum.get("material_subtypes", ""),
                cross.get("atlas_sample_type", ""),
                cross.get("atlas_lithology", ""),
                cross.get("atlas_description", ""),
                chemistry_phase,
                chemistry_row.get("additional_notes", ""),
            ]

            comments = unique_join(
                [
                    ("RELAB specimen", specimen_name),
                    ("RELAB description", spectrum.get("specimen_description", "")),
                    ("Collection", spectrum.get("specimen_collection_location", "")),
                    ("Specimen type", spectrum.get("specimen_type", "")),
                    ("Material origin", spectrum.get("material_origin", "")),
                    ("Material type", spectrum.get("material_type", "")),
                    ("Material subtypes", spectrum.get("material_subtypes", "")),
                    ("Atlas type", cross.get("atlas_sample_type", "")),
                    ("Atlas lithology", cross.get("atlas_lithology", "")),
                    ("Atlas description", cross.get("atlas_description", "")),
                    ("Chemistry phase", chemistry_phase),
                    ("Chemistry notes", chemistry_row.get("additional_notes", "")),
                    ("Parent source", parent_source),
                ]
            )

            output_rows.append(
                {
                    "id": 0,  # assigned after sorting
                    "i_atlas_sample": atlas_parent if atlas_parent is not None else -99,
                    "j_child": child or "0",
                    "k_feo_wt_percent": feo,
                    "feo_known": "yes" if known else "no",
                    "relab_measurement_id": measurement_id,
                    "relab_specimen_id": spectrum_specimen_id,
                    "relab_specimen_name": specimen_name,
                    "feo_association": association,
                    "chemistry_product_id": clean(
                        chemistry_row.get("chemistry_product_id", "")
                    ),
                    "chemistry_specimen_id": clean(chemistry_row.get("specimen_id", "")),
                    "chemistry_phase": chemistry_phase,
                    "chemistry_source_and_method": clean(
                        chemistry_row.get("sources_and_methods", "")
                    ),
                    "atlas_sample_type": clean(cross.get("atlas_sample_type", "")),
                    "atlas_lithology": clean(cross.get("atlas_lithology", "")),
                    "age_or_treatment_notes": age_treatment_notes(descriptive_values),
                    "comments": comments,
                    "crosswalk_match_type": clean(cross.get("match_type", "")),
                    "crosswalk_needs_review": clean(cross.get("needs_review", "")),
                    "atlas_detail_url": clean(cross.get("atlas_detail_url", "")),
                    "atlas_pdf_url": clean(cross.get("atlas_pdf_url", "")),
                }
            )
            stats["rows"] += 1
            stats["rows_feo_known" if known else "rows_feo_unknown"] += 1
            stats[f"association_{association}"] += 1

    def row_sort_key(row: dict[str, object]) -> tuple:
        parent = int(row["i_atlas_sample"])
        parent_key = (1, 0) if parent == -99 else (0, parent)
        child_key = child_sort_key(str(row["j_child"]))
        feo = float(row["k_feo_wt_percent"])
        return (
            parent_key,
            child_key,
            feo,
            str(row["relab_measurement_id"]),
            str(row["chemistry_product_id"]),
            str(row["chemistry_phase"]),
        )

    output_rows.sort(key=row_sort_key)
    for row_id, row in enumerate(output_rows, 1):
        row["id"] = row_id

    stats["spectra_input"] = len(spectra)
    stats["crosswalk_rows_input"] = len(crosswalk)
    stats["chemistry_rows_input"] = len(chemistry)
    return output_rows, stats


def write_output(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join RELAB spectra, the RELAB-Atlas crosswalk and RELAB chemistry; "
            "write rows sorted by Apollo parent i, child j, then FeO k."
        )
    )
    parser.add_argument("--spectrum-index", type=Path, required=True)
    parser.add_argument("--crosswalk", type=Path, required=True)
    parser.add_argument("--chemistry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--unknown-feo",
        type=float,
        default=-99.0,
        help="Sentinel written when FeO is unavailable (default: -99).",
    )
    parser.add_argument(
        "--exact-specimen-match",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use exact RELAB specimen-ID chemistry matches in addition to explicit "
            "label references (default: enabled)."
        ),
    )
    parser.add_argument(
        "--include-unmatched",
        action="store_true",
        help=(
            "Also include spectra without an Atlas match. Their i value is -99, "
            "unless an Apollo parent number exists but is missing from the Atlas."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for path in (args.spectrum_index, args.crosswalk, args.chemistry):
        if not path.is_file():
            print(f"ERROR: input file not found: {path}", file=sys.stderr)
            return 2
    try:
        rows, stats = build_rows(args)
        write_output(args.output, rows)
    except (OSError, ValueError, csv.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Output: {args.output}")
    print(f"Input spectra: {stats['spectra_input']}")
    print(f"Output rows: {stats['rows']}")
    print(f"Rows with known FeO: {stats['rows_feo_known']}")
    print(f"Rows with unknown FeO (-99): {stats['rows_feo_unknown']}")
    print(f"Explicit chemistry associations: {stats['association_explicit_reference']}")
    print(f"Exact specimen-ID associations: {stats['association_exact_specimen_id']}")
    print(f"Spectra skipped without Atlas sample: {stats['spectra_without_atlas_sample']}")
    print(f"Spectra skipped without crosswalk row: {stats['spectra_without_crosswalk']}")
    if stats["explicit_references_not_downloaded"]:
        print(
            "WARNING: explicit chemistry references absent from downloaded chemistry: "
            f"{stats['explicit_references_not_downloaded']}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
