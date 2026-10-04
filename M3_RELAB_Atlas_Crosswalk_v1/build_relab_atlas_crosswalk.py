#!/usr/bin/env python3
"""Build RELAB measurement metadata and an auditable Atlas crosswalk.

Input may be a directory containing downloaded RELAB PDS4 XML labels or a
CSV/TSV export.  Matching uses specimen names and Apollo parent sample numbers;
spectral values are not needed at this stage.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


# Apollo parent-sample numbers are five digits.  Apollo 11/12/14/15 numbers
# start with the mission number; Apollo 16 and 17 numbers start with 6 and 7.
# Restricting the pattern prevents arbitrary five-digit numbers (for example a
# Luna catalogue number) from silently becoming Apollo samples.
APOLLO_PARENT_PATTERN = r"(?:10\d{3}|12\d{3}|14\d{3}|15\d{3}|6\d{4}|7\d{4})"
PARENT_RE = re.compile(rf"(?<!\d)({APOLLO_PARENT_PATTERN})(?!\d)")

# Some RELAB names use thousands separators, e.g. 10,003,173 means Apollo
# parent 10003, subsample 173.  This must be handled before PARENT_RE.
GROUPED_SAMPLE_RE = re.compile(
    r"(?<!\d)(1[0245]|[67]\d)\s*,\s*(\d{3})(?:\s*,\s*([A-Za-z0-9_-]+))?(?!\d)"
)
SUBSAMPLE_AFTER_PARENT_RE = re.compile(r"^\s*[,.;]\s*([A-Za-z0-9_-]+)")
NON_APOLLO_RE = re.compile(r"\b(?:luna|lunokhod)\b", re.IGNORECASE)

MEASUREMENT_FIELDS = [
    "measurement_id",
    "logical_identifier",
    "version_id",
    "title",
    "specimen_id",
    "specimen_name",
    "specimen_description",
    "specimen_collection_location",
    "specimen_type",
    "material_origin",
    "material_type",
    "material_subtypes",
    "spectral_range_min",
    "spectral_range_max",
    "spectral_range_unit",
    "incidence_angle_deg",
    "emission_angle_deg",
    "phase_angle_deg",
    "measurement_data_file",
    "source_metadata_file",
]


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def clean(value: str | None) -> str:
    return " ".join((value or "").replace("\xa0", " ").split())


def first_text(elements: dict[str, list[ET.Element]], name: str) -> str:
    for element in elements.get(name, []):
        value = clean(element.text)
        if value:
            return value
    return ""


def all_text(elements: dict[str, list[ET.Element]], name: str) -> str:
    values: list[str] = []
    for element in elements.get(name, []):
        value = clean(element.text)
        if value and value not in values:
            values.append(value)
    return "|".join(values)


def parse_relab_xml(path: Path) -> dict[str, str] | None:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return None

    elements: dict[str, list[ET.Element]] = {}
    for element in root.iter():
        elements.setdefault(local_name(element.tag), []).append(element)

    logical_identifier = first_text(elements, "logical_identifier")
    specimen_name = first_text(elements, "specimen_name")
    spectral_min = first_text(elements, "spectral_range_min")
    spectral_max = first_text(elements, "spectral_range_max")

    # Ignore chemistry labels, collection inventories, and unrelated XML.
    is_reflectance = "data_reflectance" in logical_identifier
    if not (is_reflectance and specimen_name and spectral_min and spectral_max):
        return None

    measurement_id = logical_identifier.rstrip(":").rsplit(":", 1)[-1]
    data_files = [clean(e.text) for e in elements.get("file_name", []) if clean(e.text)]
    tab_file = next((name for name in data_files if name.lower().endswith(".tab")), "")

    def angle(name: str) -> str:
        return first_text(elements, name)

    return {
        "measurement_id": measurement_id or path.stem,
        "logical_identifier": logical_identifier,
        "version_id": first_text(elements, "version_id"),
        "title": first_text(elements, "title"),
        "specimen_id": first_text(elements, "specimen_id"),
        "specimen_name": specimen_name,
        "specimen_description": first_text(elements, "specimen_description"),
        "specimen_collection_location": first_text(elements, "specimen_collection_location"),
        "specimen_type": first_text(elements, "specimen_type"),
        "material_origin": first_text(elements, "material_origin"),
        "material_type": first_text(elements, "material_type"),
        "material_subtypes": all_text(elements, "material_subtype"),
        "spectral_range_min": spectral_min,
        "spectral_range_max": spectral_max,
        "spectral_range_unit": first_text(elements, "spectral_range_unit_name"),
        "incidence_angle_deg": angle("incidence_angle"),
        "emission_angle_deg": angle("emission_angle"),
        "phase_angle_deg": angle("phase_angle"),
        "measurement_data_file": tab_file,
        "source_metadata_file": str(path.resolve()),
    }


def load_xml_directory(directory: Path) -> tuple[list[dict[str, str]], list[str]]:
    records: list[dict[str, str]] = []
    warnings: list[str] = []
    xml_files = sorted(directory.rglob("*.xml"))
    if not xml_files:
        raise ValueError(f"No XML files found under {directory}")
    for path in xml_files:
        record = parse_relab_xml(path)
        if record:
            records.append(record)
    if not records:
        raise ValueError(
            "XML files were found, but none looked like RELAB reflectance measurement labels."
        )
    duplicate_ids = [key for key, count in Counter(r["measurement_id"] for r in records).items() if count > 1]
    if duplicate_ids:
        warnings.append(f"Duplicate measurement IDs found: {', '.join(duplicate_ids[:20])}")
    return records, warnings


def normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


CSV_ALIASES = {
    "measurement_id": ("measurementid", "spectrumid", "measurement"),
    "logical_identifier": ("logicalidentifier", "lid", "pds4productlid"),
    "specimen_id": ("specimenid",),
    "specimen_name": ("specimenname", "samplename", "sampleid", "samplenumber"),
    "specimen_description": ("specimendescription", "description"),
    "specimen_collection_location": ("specimencollectionlocation", "collectionlocation"),
    "specimen_type": ("specimentype",),
    "material_origin": ("materialorigin",),
    "material_type": ("materialtype",),
    "material_subtypes": ("materialsubtypes", "materialsubtype"),
    "spectral_range_min": ("spectralrangemin", "wavelengthmin", "minwavelength"),
    "spectral_range_max": ("spectralrangemax", "wavelengthmax", "maxwavelength"),
    "spectral_range_unit": ("spectralrangeunit", "wavelengthunit"),
    "incidence_angle_deg": ("incidenceangle", "incidenceangledeg"),
    "emission_angle_deg": ("emissionangle", "emissionangledeg"),
    "phase_angle_deg": ("phaseangle", "phaseangledeg"),
    "measurement_data_file": ("measurementdatafile", "spectrumfile", "datafile"),
}


def load_relab_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    sample = path.read_text(encoding="utf-8-sig", errors="replace")[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
    except csv.Error:
        dialect = csv.excel_tab if path.suffix.lower() == ".tsv" else csv.excel

    records: list[dict[str, str]] = []
    warnings: list[str] = []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, dialect=dialect)
        if not reader.fieldnames:
            raise ValueError(f"No header row found in {path}")
        header_map = {normalized_key(name): name for name in reader.fieldnames}

        selected: dict[str, str] = {}
        for target, aliases in CSV_ALIASES.items():
            for alias in aliases:
                if alias in header_map:
                    selected[target] = header_map[alias]
                    break
        if "specimen_name" not in selected:
            raise ValueError(
                "RELAB CSV needs a specimen-name/sample-number column. Recognized examples: "
                "specimen_name, Specimen Name, sample_id, sample_number."
            )

        for row_number, source in enumerate(reader, start=2):
            record = {field: "" for field in MEASUREMENT_FIELDS}
            for target, source_name in selected.items():
                record[target] = clean(source.get(source_name))
            if not record["measurement_id"]:
                record["measurement_id"] = f"csv-row-{row_number:06d}"
            record["source_metadata_file"] = f"{path.resolve()}#row={row_number}"
            records.append(record)

    if not records:
        raise ValueError(f"No data rows found in {path}")
    return records, warnings


def read_atlas_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"atlas_sample_id", "sample_type", "lithology", "description", "detail_url"}
    missing = required.difference(rows[0].keys() if rows else set())
    if missing:
        raise ValueError(f"Atlas CSV is missing columns: {', '.join(sorted(missing))}")
    return rows


def parse_sample_reference(
    specimen_name: str, specimen_collection_location: str = ""
) -> dict[str, str]:
    """Conservatively parse the lunar mission and Apollo sample reference.

    This parser identifies links to Atlas *parent-sample pages*.  It does not
    claim that a RELAB split, grain-size fraction, mineral separate, or other
    prepared material has the parent's reported chemical composition.
    """

    name = clean(specimen_name)
    context = clean(f"{specimen_name} {specimen_collection_location}")
    if NON_APOLLO_RE.search(context):
        return {
            "scope": "non_apollo",
            "parent": "",
            "normalized": "",
            "has_subsample_or_context": "no",
        }

    grouped = GROUPED_SAMPLE_RE.search(name)
    if grouped:
        parent = grouped.group(1) + grouped.group(2)
        suffix = clean(grouped.group(3))
        normalized = parent + (f",{suffix}" if suffix else "")
        remaining = clean(name[: grouped.start()] + " " + name[grouped.end() :])
        return {
            "scope": "apollo",
            "parent": parent,
            "normalized": normalized,
            "has_subsample_or_context": "yes" if suffix or remaining else "no",
        }

    match = PARENT_RE.search(name)
    if not match:
        return {
            "scope": "unparsed",
            "parent": "",
            "normalized": "",
            "has_subsample_or_context": "no",
        }

    parent = match.group(1)
    tail = name[match.end() :]
    suffix_match = SUBSAMPLE_AFTER_PARENT_RE.match(tail)
    suffix = clean(suffix_match.group(1)) if suffix_match else ""
    if suffix_match:
        tail = tail[suffix_match.end() :]
    remaining = clean(name[: match.start()] + " " + tail)
    normalized = parent + (f",{suffix}" if suffix else "")
    return {
        "scope": "apollo",
        "parent": parent,
        "normalized": normalized,
        "has_subsample_or_context": "yes" if suffix or remaining else "no",
    }


def normalize_specimen_name(value: str) -> str:
    """Backward-compatible helper returning the parsed sample reference."""

    return parse_sample_reference(value)["normalized"]


def parent_sample(value: str) -> str:
    """Backward-compatible helper returning the parsed Apollo parent ID."""

    return parse_sample_reference(value)["parent"]


def build_crosswalk(
    measurements: list[dict[str, str]], atlas_rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    atlas_by_id: dict[str, list[dict[str, str]]] = {}
    for row in atlas_rows:
        atlas_by_id.setdefault(clean(row.get("atlas_sample_id")), []).append(row)

    results: list[dict[str, str]] = []
    for measurement in measurements:
        specimen_name = clean(measurement.get("specimen_name"))
        parsed = parse_sample_reference(
            specimen_name, clean(measurement.get("specimen_collection_location"))
        )
        normalized = parsed["normalized"]
        parent = parsed["parent"]
        candidates = atlas_by_id.get(parent, []) if parent else []

        if parsed["scope"] == "non_apollo":
            match_type = "non_apollo"
            reason = "RELAB identifies this as a non-Apollo lunar sample; Apollo Atlas matching is not applicable"
            atlas = {}
        elif not parent:
            match_type = "unparsed"
            reason = "No recognizable Apollo parent-sample number found in the RELAB specimen name"
            atlas = {}
        elif not candidates:
            match_type = "atlas_missing"
            reason = f"Recognizable Apollo parent sample {parent} is absent from the local Atlas index"
            atlas = {}
        elif len(candidates) > 1:
            match_type = "ambiguous"
            reason = f"Atlas contains {len(candidates)} rows for parent sample {parent}"
            atlas = {}
        else:
            atlas = candidates[0]
            if parsed["has_subsample_or_context"] == "no":
                match_type = "sample_number"
                reason = "The complete RELAB specimen name is the same five-digit Apollo sample number"
            else:
                match_type = "subsample_or_fraction"
                reason = (
                    f"RELAB material {specimen_name} linked only to Atlas parent page {parent}; "
                    "composition of this subsample, size fraction, mineral separate, or preparation is not implied"
                )

        if match_type == "sample_number":
            composition_note = (
                "Atlas parent-sample link established; composition values still require source-level verification"
            )
        elif match_type == "subsample_or_fraction":
            composition_note = "Do not transfer parent-sample composition without subsample-specific verification"
        elif match_type == "non_apollo":
            composition_note = "Outside the Apollo Atlas crosswalk"
        else:
            composition_note = "No composition link established"

        results.append(
            {
                "measurement_id": measurement.get("measurement_id", ""),
                "relab_specimen_id": measurement.get("specimen_id", ""),
                "relab_specimen_name": specimen_name,
                "normalized_specimen_name": normalized,
                "apollo_parent_sample": parent,
                "atlas_sample_id": atlas.get("atlas_sample_id", ""),
                "match_type": match_type,
                "match_reason": reason,
                "composition_transfer_note": composition_note,
                "atlas_sample_type": atlas.get("sample_type", ""),
                "atlas_lithology": atlas.get("lithology", ""),
                "atlas_description": atlas.get("description", ""),
                "atlas_detail_url": atlas.get("detail_url", ""),
                "atlas_pdf_url": atlas.get("atlas_pdf_url", ""),
                "needs_review": "no"
                if match_type in {"sample_number", "non_apollo"}
                else "yes",
            }
        )
    return results


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-csv", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--relab-dir", type=Path, help="Directory containing RELAB PDS4 XML labels")
    source.add_argument("--relab-csv", type=Path, help="RELAB CSV/TSV metadata export")
    parser.add_argument("--output-dir", type=Path, default=Path("crosswalk_output"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        atlas_rows = read_atlas_csv(args.atlas_csv)
        if args.relab_dir:
            measurements, warnings = load_xml_directory(args.relab_dir)
            source = str(args.relab_dir.resolve())
        else:
            measurements, warnings = load_relab_csv(args.relab_csv)
            source = str(args.relab_csv.resolve())

        crosswalk = build_crosswalk(measurements, atlas_rows)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        measurements_path = args.output_dir / "relab_measurements.csv"
        crosswalk_path = args.output_dir / "relab_atlas_crosswalk.csv"
        summary_path = args.output_dir / "crosswalk_summary.json"
        write_rows(measurements_path, MEASUREMENT_FIELDS, measurements)
        crosswalk_fields = list(crosswalk[0].keys()) if crosswalk else []
        write_rows(crosswalk_path, crosswalk_fields, crosswalk)

        counts = Counter(row["match_type"] for row in crosswalk)
        summary = {
            "schema": "relab-atlas-crosswalk-summary-2.0",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "atlas_csv": str(args.atlas_csv.resolve()),
            "relab_source": source,
            "measurement_count": len(measurements),
            "match_counts": dict(sorted(counts.items())),
            "warnings": warnings,
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

        print(f"RELAB measurements written: {measurements_path} ({len(measurements)} rows)")
        print(f"Crosswalk written: {crosswalk_path}")
        print("Match summary: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))
        for warning in warnings:
            print(f"WARNING: {warning}", file=sys.stderr)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
