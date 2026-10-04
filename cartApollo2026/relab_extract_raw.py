#!/usr/bin/env python3
"""Extract native-grid RELAB spectra and specimen chemistry from PDS4 products.

The extractor is deliberately a raw-data stage.  It does not interpolate,
resample, smooth, normalize, or map spectra to M3 bands.  PDS4 XML labels are
authoritative for byte offsets, record counts, field positions, and units.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median


POINT_FIELDS = [
    "measurement_id",
    "point_index",
    "wavelength_original",
    "wavelength_unit_original",
    "wavelength_nm",
    "reflectance",
    "uncertainty_value",
    "uncertainty_type",
    "extra_fields_json",
]

INDEX_FIELDS = [
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
    "instrument_names",
    "measurement_segments",
    "segment_numbers",
    "incidence_angle_deg",
    "emission_angle_deg",
    "phase_angle_deg",
    "label_spectral_range_min",
    "label_spectral_range_max",
    "label_spectral_range_unit",
    "native_point_count",
    "native_field_count",
    "native_field_names",
    "standard_deviation_present",
    "data_wavelength_min_nm",
    "data_wavelength_max_nm",
    "wavelength_strictly_increasing",
    "wavelength_uniform_step",
    "wavelength_median_step_nm",
    "wavelength_min_step_nm",
    "wavelength_max_step_nm",
    "chemistry_lid_references",
    "measurement_data_file",
    "source_metadata_file",
    "measurement_data_sha256",
    "source_metadata_sha256",
]

CHEMISTRY_PREFIX_FIELDS = [
    "chemistry_product_id",
    "logical_identifier",
    "version_id",
    "chemistry_record_index",
]

CHEMISTRY_SUFFIX_FIELDS = [
    "source_column_names_json",
    "chemistry_data_file",
    "source_metadata_file",
    "chemistry_data_sha256",
    "source_metadata_sha256",
]


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def clean(value: str | None) -> str:
    return " ".join((value or "").replace("\xa0", " ").split())


def children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element if local_name(child.tag) == name]


def first_child(element: ET.Element, name: str) -> ET.Element | None:
    return next((child for child in element if local_name(child.tag) == name), None)


def descendants(element: ET.Element, name: str) -> list[ET.Element]:
    return [item for item in element.iter() if local_name(item.tag) == name]


def first_text(root: ET.Element, name: str) -> str:
    for element in descendants(root, name):
        value = clean(element.text)
        if value:
            return value
    return ""


def all_text(root: ET.Element, name: str) -> list[str]:
    values: list[str] = []
    for element in descendants(root, name):
        value = clean(element.text)
        if value and value not in values:
            values.append(value)
    return values


def direct_text(element: ET.Element, name: str) -> str:
    item = first_child(element, name)
    return clean(item.text) if item is not None else ""


def normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def canonical_column_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", clean(value)).strip("_").lower()
    text = re.sub(r"_+", "_", text)
    aliases = {
        "phase": "phase_name",
        "feo_percent": "feo_percent",
        "sources_and_methods": "sources_and_methods",
        "additional_notes": "additional_notes",
    }
    return aliases.get(text, text)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_float(value: str, context: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"Expected a real number for {context}, found {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Non-finite value for {context}: {value!r}")
    return result


def format_number(value: float | None) -> str:
    return "" if value is None else format(value, ".17g")


def wavelength_to_nm(value: float, unit: str) -> float:
    key = normalized_name(unit)
    if key in {"nm", "nanometer", "nanometers", "nanometre", "nanometres"}:
        return value
    if key in {"um", "micron", "microns", "micrometer", "micrometers", "micrometre", "micrometres"}:
        return value * 1000.0
    if key in {"angstrom", "angstroms", "a"}:
        return value * 0.1
    raise ValueError(f"Unsupported wavelength unit {unit!r}")


def read_xml(path: Path) -> ET.Element:
    try:
        return ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Invalid XML label {path}: {exc}") from exc


def reflectance_label_info(path: Path) -> dict[str, object] | None:
    root = read_xml(path)
    logical_identifier = first_text(root, "logical_identifier")
    if "data_reflectance" not in logical_identifier:
        return None

    file_areas = descendants(root, "File_Area_Observational")
    if not file_areas:
        raise ValueError(f"No File_Area_Observational in {path}")
    file_area = file_areas[0]
    file_elements = children(file_area, "File")
    table_elements = children(file_area, "Table_Character")
    if len(file_elements) != 1 or len(table_elements) != 1:
        raise ValueError(
            f"Expected one File and one Table_Character in {path}; "
            f"found {len(file_elements)} and {len(table_elements)}"
        )

    data_name = direct_text(file_elements[0], "file_name")
    if not data_name:
        raise ValueError(f"No data file_name in {path}")
    data_path = path.parent / data_name
    if not data_path.is_file():
        raise ValueError(f"Data file declared by {path} is missing: {data_path}")

    table = table_elements[0]
    offset_text = direct_text(table, "offset")
    records_text = direct_text(table, "records")
    record_elements = children(table, "Record_Character")
    if len(record_elements) != 1:
        raise ValueError(f"Expected one Record_Character in {path}")
    record = record_elements[0]
    record_length_text = direct_text(record, "record_length")
    declared_fields_text = direct_text(record, "fields")
    if not all([offset_text, records_text, record_length_text, declared_fields_text]):
        raise ValueError(f"Incomplete table dimensions in {path}")

    field_definitions: list[dict[str, object]] = []
    for field in children(record, "Field_Character"):
        field_definitions.append(
            {
                "name": direct_text(field, "name"),
                "number": int(direct_text(field, "field_number")),
                "location": int(direct_text(field, "field_location")),
                "length": int(direct_text(field, "field_length")),
                "data_type": direct_text(field, "data_type"),
                "unit": direct_text(field, "unit"),
            }
        )
    field_definitions.sort(key=lambda item: int(item["number"]))
    declared_fields = int(declared_fields_text)
    if declared_fields != len(field_definitions):
        raise ValueError(
            f"Field count mismatch in {path}: label says {declared_fields}, "
            f"but {len(field_definitions)} definitions were found"
        )

    measurement_id = logical_identifier.rstrip(":").rsplit(":", 1)[-1]
    instruments = all_text(root, "instrument_name")
    chemistry_refs: list[str] = []
    for reference in descendants(root, "Internal_Reference"):
        reference_type = direct_text(reference, "reference_type")
        lid = direct_text(reference, "lid_reference")
        if reference_type == "data_to_ancillary_data" and "chemistry" in lid.lower():
            chemistry_refs.append(lid)

    return {
        "root": root,
        "measurement_id": measurement_id or path.stem,
        "logical_identifier": logical_identifier,
        "version_id": first_text(root, "version_id"),
        "title": first_text(root, "title"),
        "specimen_id": first_text(root, "specimen_id"),
        "specimen_name": first_text(root, "specimen_name"),
        "specimen_description": first_text(root, "specimen_description"),
        "specimen_collection_location": first_text(root, "specimen_collection_location"),
        "specimen_type": first_text(root, "specimen_type"),
        "material_origin": first_text(root, "material_origin"),
        "material_type": first_text(root, "material_type"),
        "material_subtypes": all_text(root, "material_subtype"),
        "instrument_names": instruments,
        "measurement_segments": first_text(root, "measurement_segments"),
        "segment_numbers": all_text(root, "segment_number"),
        "incidence_angle_deg": first_text(root, "incidence_angle"),
        "emission_angle_deg": first_text(root, "emission_angle"),
        "phase_angle_deg": first_text(root, "phase_angle"),
        "label_spectral_range_min": first_text(root, "spectral_range_min"),
        "label_spectral_range_max": first_text(root, "spectral_range_max"),
        "label_spectral_range_unit": first_text(root, "spectral_range_unit_name"),
        "chemistry_lid_references": chemistry_refs,
        "data_path": data_path,
        "label_path": path,
        "offset": int(offset_text),
        "records": int(records_text),
        "record_length": int(record_length_text),
        "field_definitions": field_definitions,
    }


def identify_spectral_fields(field_definitions: list[dict[str, object]]) -> tuple[int, int, int | None]:
    wavelength_index: int | None = None
    reflectance_index: int | None = None
    uncertainty_index: int | None = None
    for index, field in enumerate(field_definitions):
        key = normalized_name(str(field["name"]))
        if "wavelength" in key:
            wavelength_index = index
        elif key == "reflectance" or "reflectance" in key:
            reflectance_index = index
        elif any(word in key for word in ("standarddeviation", "standarderror", "uncertainty")):
            uncertainty_index = index
    if wavelength_index is None or reflectance_index is None:
        names = [str(field["name"]) for field in field_definitions]
        raise ValueError(f"Could not identify wavelength and reflectance fields among {names}")
    return wavelength_index, reflectance_index, uncertainty_index


def extract_spectrum(info: dict[str, object], hash_inputs: bool) -> tuple[dict[str, str], list[dict[str, str]], list[str]]:
    data_path = Path(info["data_path"])
    label_path = Path(info["label_path"])
    payload = data_path.read_bytes()
    offset = int(info["offset"])
    records = int(info["records"])
    record_length = int(info["record_length"])
    required_bytes = offset + records * record_length
    if len(payload) < required_bytes:
        raise ValueError(
            f"{data_path} is too short: {len(payload)} bytes, "
            f"but label requires at least {required_bytes}"
        )

    field_definitions = list(info["field_definitions"])
    wavelength_index, reflectance_index, uncertainty_index = identify_spectral_fields(field_definitions)
    wavelength_field = field_definitions[wavelength_index]
    wavelength_unit = clean(str(wavelength_field.get("unit", ""))) or clean(
        str(info["label_spectral_range_unit"])
    )
    if not wavelength_unit:
        raise ValueError(f"No wavelength unit in {label_path}")

    points: list[dict[str, str]] = []
    wavelengths_nm: list[float] = []
    warnings: list[str] = []

    for point_index in range(records):
        start = offset + point_index * record_length
        record = payload[start : start + record_length]
        values: list[str] = []
        for field in field_definitions:
            location = int(field["location"]) - 1
            length = int(field["length"])
            raw = record[location : location + length]
            try:
                values.append(raw.decode("ascii").strip())
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"Non-ASCII data in {data_path}, record {point_index + 1}, "
                    f"field {field['name']}"
                ) from exc

        wavelength_original = parse_float(
            values[wavelength_index], f"{data_path} record {point_index + 1} wavelength"
        )
        reflectance = parse_float(
            values[reflectance_index], f"{data_path} record {point_index + 1} reflectance"
        )
        wavelength_nm = wavelength_to_nm(wavelength_original, wavelength_unit)
        wavelengths_nm.append(wavelength_nm)

        uncertainty_value = ""
        uncertainty_type = ""
        if uncertainty_index is not None:
            uncertainty_type = str(field_definitions[uncertainty_index]["name"])
            if values[uncertainty_index]:
                parse_float(
                    values[uncertainty_index],
                    f"{data_path} record {point_index + 1} {uncertainty_type}",
                )
                uncertainty_value = values[uncertainty_index]

        reserved = {wavelength_index, reflectance_index}
        if uncertainty_index is not None:
            reserved.add(uncertainty_index)
        extras = {
            str(field_definitions[index]["name"]): values[index]
            for index in range(len(field_definitions))
            if index not in reserved
        }
        points.append(
            {
                "measurement_id": str(info["measurement_id"]),
                "point_index": str(point_index + 1),
                "wavelength_original": values[wavelength_index],
                "wavelength_unit_original": wavelength_unit,
                "wavelength_nm": format_number(wavelength_nm),
                "reflectance": values[reflectance_index],
                "uncertainty_value": uncertainty_value,
                "uncertainty_type": uncertainty_type,
                "extra_fields_json": json.dumps(extras, sort_keys=True) if extras else "",
            }
        )

    steps = [b - a for a, b in zip(wavelengths_nm, wavelengths_nm[1:])]
    strictly_increasing = all(step > 0 for step in steps)
    if not strictly_increasing:
        warnings.append(f"{info['measurement_id']}: wavelength grid is not strictly increasing")
    if steps:
        representative_step = median(steps)
        tolerance = max(1e-10, abs(representative_step) * 1e-9)
        uniform_step = all(abs(step - representative_step) <= tolerance for step in steps)
        min_step = min(steps)
        max_step = max(steps)
    else:
        representative_step = min_step = max_step = None
        uniform_step = True

    label_min = clean(str(info["label_spectral_range_min"]))
    label_max = clean(str(info["label_spectral_range_max"]))
    label_unit = clean(str(info["label_spectral_range_unit"]))
    if label_min and label_max and label_unit:
        expected_min = wavelength_to_nm(parse_float(label_min, "label range minimum"), label_unit)
        expected_max = wavelength_to_nm(parse_float(label_max, "label range maximum"), label_unit)
        endpoint_tolerance = max(1e-8, abs(representative_step or 0.0) * 1e-6)
        if abs(wavelengths_nm[0] - expected_min) > endpoint_tolerance:
            warnings.append(
                f"{info['measurement_id']}: data minimum {wavelengths_nm[0]} nm "
                f"differs from label minimum {expected_min} nm"
            )
        if abs(wavelengths_nm[-1] - expected_max) > endpoint_tolerance:
            warnings.append(
                f"{info['measurement_id']}: data maximum {wavelengths_nm[-1]} nm "
                f"differs from label maximum {expected_max} nm"
            )

    field_names = [str(field["name"]) for field in field_definitions]
    index_row = {
        "measurement_id": str(info["measurement_id"]),
        "logical_identifier": str(info["logical_identifier"]),
        "version_id": str(info["version_id"]),
        "title": str(info["title"]),
        "specimen_id": str(info["specimen_id"]),
        "specimen_name": str(info["specimen_name"]),
        "specimen_description": str(info["specimen_description"]),
        "specimen_collection_location": str(info["specimen_collection_location"]),
        "specimen_type": str(info["specimen_type"]),
        "material_origin": str(info["material_origin"]),
        "material_type": str(info["material_type"]),
        "material_subtypes": "|".join(info["material_subtypes"]),
        "instrument_names": "|".join(info["instrument_names"]),
        "measurement_segments": str(info["measurement_segments"]),
        "segment_numbers": "|".join(info["segment_numbers"]),
        "incidence_angle_deg": str(info["incidence_angle_deg"]),
        "emission_angle_deg": str(info["emission_angle_deg"]),
        "phase_angle_deg": str(info["phase_angle_deg"]),
        "label_spectral_range_min": label_min,
        "label_spectral_range_max": label_max,
        "label_spectral_range_unit": label_unit,
        "native_point_count": str(records),
        "native_field_count": str(len(field_definitions)),
        "native_field_names": "|".join(field_names),
        "standard_deviation_present": "yes"
        if uncertainty_index is not None and "standarddeviation" in normalized_name(field_names[uncertainty_index])
        else "no",
        "data_wavelength_min_nm": format_number(wavelengths_nm[0]),
        "data_wavelength_max_nm": format_number(wavelengths_nm[-1]),
        "wavelength_strictly_increasing": "yes" if strictly_increasing else "no",
        "wavelength_uniform_step": "yes" if uniform_step else "no",
        "wavelength_median_step_nm": format_number(representative_step),
        "wavelength_min_step_nm": format_number(min_step),
        "wavelength_max_step_nm": format_number(max_step),
        "chemistry_lid_references": "|".join(info["chemistry_lid_references"]),
        "measurement_data_file": str(data_path.resolve()),
        "source_metadata_file": str(label_path.resolve()),
        "measurement_data_sha256": sha256_file(data_path) if hash_inputs else "",
        "source_metadata_sha256": sha256_file(label_path) if hash_inputs else "",
    }
    return index_row, points, warnings


def chemistry_label_info(path: Path) -> dict[str, object] | None:
    root = read_xml(path)
    logical_identifier = first_text(root, "logical_identifier")
    if "data_ancillary_chemistry" not in logical_identifier:
        return None
    file_areas = descendants(root, "File_Area_Ancillary")
    if not file_areas:
        raise ValueError(f"No File_Area_Ancillary in {path}")
    file_area = file_areas[0]
    file_elements = children(file_area, "File")
    if len(file_elements) != 1:
        raise ValueError(f"Expected one chemistry File in {path}")
    data_name = direct_text(file_elements[0], "file_name")
    data_path = path.parent / data_name
    if not data_path.is_file():
        raise ValueError(f"Chemistry CSV declared by {path} is missing: {data_path}")

    chemistry_table: ET.Element | None = None
    for table in children(file_area, "Table_Delimited"):
        if direct_text(table, "local_identifier") == "Chemistry-Data":
            chemistry_table = table
            break
    if chemistry_table is None:
        raise ValueError(f"No Chemistry-Data table in {path}")
    record_elements = children(chemistry_table, "Record_Delimited")
    if len(record_elements) != 1:
        raise ValueError(f"Expected one Chemistry-Data Record_Delimited in {path}")
    field_elements = children(record_elements[0], "Field_Delimited")
    field_elements.sort(key=lambda item: int(direct_text(item, "field_number")))
    field_names = [direct_text(item, "name") for item in field_elements]
    declared_records = int(direct_text(chemistry_table, "records"))
    product_id = logical_identifier.rstrip(":").rsplit(":", 1)[-1]
    return {
        "logical_identifier": logical_identifier,
        "version_id": first_text(root, "version_id"),
        "product_id": product_id or path.stem,
        "data_path": data_path,
        "label_path": path,
        "field_names": field_names,
        "declared_records": declared_records,
    }


def extract_chemistry(info: dict[str, object], hash_inputs: bool) -> tuple[list[dict[str, str]], list[str]]:
    data_path = Path(info["data_path"])
    label_path = Path(info["label_path"])
    with data_path.open("r", encoding="utf-8-sig", errors="strict", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty chemistry CSV {data_path}")
    source_headers = rows[0]
    label_names = list(info["field_names"])
    if len(source_headers) != len(label_names):
        raise ValueError(
            f"Chemistry column mismatch for {data_path}: CSV has {len(source_headers)}, "
            f"label defines {len(label_names)}"
        )
    data_rows = rows[1:]
    if len(data_rows) != int(info["declared_records"]):
        raise ValueError(
            f"Chemistry record mismatch for {data_path}: CSV has {len(data_rows)}, "
            f"label declares {info['declared_records']}"
        )

    canonical_names = [canonical_column_name(name) for name in label_names]
    if len(set(canonical_names)) != len(canonical_names):
        raise ValueError(f"Duplicate canonical chemistry column names in {label_path}")

    extracted: list[dict[str, str]] = []
    for record_index, values in enumerate(data_rows, start=1):
        if len(values) != len(label_names):
            raise ValueError(
                f"Chemistry row {record_index} in {data_path} has {len(values)} values; "
                f"expected {len(label_names)}"
            )
        record = {
            "chemistry_product_id": str(info["product_id"]),
            "logical_identifier": str(info["logical_identifier"]),
            "version_id": str(info["version_id"]),
            "chemistry_record_index": str(record_index),
        }
        record.update({name: clean(value) for name, value in zip(canonical_names, values)})
        record.update(
            {
                "source_column_names_json": json.dumps(source_headers),
                "chemistry_data_file": str(data_path.resolve()),
                "source_metadata_file": str(label_path.resolve()),
                "chemistry_data_sha256": sha256_file(data_path) if hash_inputs else "",
                "source_metadata_sha256": sha256_file(label_path) if hash_inputs else "",
            }
        )
        extracted.append(record)
    return extracted, canonical_names


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relab-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--measurement-ids",
        help="Optional comma-separated measurement IDs for a bounded test extraction",
    )
    parser.add_argument("--hash-inputs", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Stop at the first product error")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected_ids = (
        {value.strip().lower() for value in args.measurement_ids.split(",") if value.strip()}
        if args.measurement_ids
        else None
    )
    output_files = {
        "index": args.output_dir / "relab_spectrum_index.csv",
        "points": args.output_dir / "relab_spectral_points.csv",
        "chemistry": args.output_dir / "relab_chemistry.csv",
        "manifest": args.output_dir / "relab_raw_extraction_manifest.json",
    }
    existing = [path for path in output_files.values() if path.exists()]
    if existing and not args.overwrite:
        print(
            "ERROR: output files already exist; use a new directory or --overwrite: "
            + ", ".join(str(path) for path in existing),
            file=sys.stderr,
        )
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict[str, str]] = []
    point_rows: list[dict[str, str]] = []
    chemistry_rows: list[dict[str, str]] = []
    chemistry_columns: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []
    field_count_distribution: Counter[int] = Counter()

    xml_files = sorted(args.relab_dir.rglob("*.xml"))
    for path in xml_files:
        try:
            info = reflectance_label_info(path)
            if info is None:
                continue
            if selected_ids is not None and str(info["measurement_id"]).lower() not in selected_ids:
                continue
            index_row, points, product_warnings = extract_spectrum(info, args.hash_inputs)
            index_rows.append(index_row)
            point_rows.extend(points)
            warnings.extend(product_warnings)
            field_count_distribution[int(index_row["native_field_count"])] += 1
        except Exception as exc:
            message = f"{path}: {exc}"
            errors.append(message)
            if args.strict:
                print(f"ERROR: {message}", file=sys.stderr)
                return 2

    selected_chemistry_lids: set[str] | None = None
    if selected_ids is not None:
        selected_chemistry_lids = {
            lid
            for row in index_rows
            for lid in row["chemistry_lid_references"].split("|")
            if lid
        }

    for path in xml_files:
        try:
            info = chemistry_label_info(path)
            if info is None:
                continue
            if (
                selected_chemistry_lids is not None
                and str(info["logical_identifier"]) not in selected_chemistry_lids
            ):
                continue
            rows, columns = extract_chemistry(info, args.hash_inputs)
            chemistry_rows.extend(rows)
            for column in columns:
                if column not in chemistry_columns:
                    chemistry_columns.append(column)
        except Exception as exc:
            message = f"{path}: {exc}"
            errors.append(message)
            if args.strict:
                print(f"ERROR: {message}", file=sys.stderr)
                return 2

    if not index_rows:
        errors.append("No reflectance measurements were extracted")

    write_csv(output_files["index"], INDEX_FIELDS, index_rows)
    write_csv(output_files["points"], POINT_FIELDS, point_rows)
    chemistry_fieldnames = CHEMISTRY_PREFIX_FIELDS + chemistry_columns + CHEMISTRY_SUFFIX_FIELDS
    write_csv(output_files["chemistry"], chemistry_fieldnames, chemistry_rows)

    manifest = {
        "schema": "relab-raw-extraction-manifest-1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "native-grid extraction only; no interpolation, resampling, smoothing, or normalization",
        "relab_dir": str(args.relab_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "measurement_id_filter": sorted(selected_ids) if selected_ids is not None else None,
        "chemistry_scope": "referenced_by_selected_measurements"
        if selected_ids is not None
        else "all_downloaded_chemistry_products",
        "hash_inputs": args.hash_inputs,
        "measurement_count": len(index_rows),
        "spectral_point_count": len(point_rows),
        "native_field_count_distribution": {
            str(key): value for key, value in sorted(field_count_distribution.items())
        },
        "chemistry_record_count": len(chemistry_rows),
        "warnings": warnings,
        "errors": errors,
        "outputs": {key: str(path.resolve()) for key, path in output_files.items() if key != "manifest"},
    }
    output_files["manifest"].write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Measurements extracted: {len(index_rows)}")
    print(f"Spectral points extracted: {len(point_rows)}")
    print(
        "Native field counts: "
        + ", ".join(f"{key} columns={value}" for key, value in sorted(field_count_distribution.items()))
    )
    print(f"Chemistry records extracted: {len(chemistry_rows)}")
    print(f"Manifest: {output_files['manifest']}")
    if warnings:
        print(f"Warnings: {len(warnings)} (see manifest)")
    if errors:
        print(f"Errors: {len(errors)} (see manifest)", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
