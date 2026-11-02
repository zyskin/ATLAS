#!/usr/bin/env python3
"""Combine traceable source extracts into one Apollo-sample chemistry index.

This is an index, not a conflict-resolution step.  Measurements from different
source documents remain separate rows.  No parent-to-child transfer, averaging,
or imputation is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "sample_id",
    "material_type",
    "source_dataset",
    "feo_wt_percent",
    "is_feo",
    "al2o3_wt_percent",
    "tio2_wt_percent",
    "mgo_wt_percent",
    "cao_wt_percent",
    "source_viewer_page",
    "source_printed_page",
    "table_reference",
    "source_url",
    "extraction_status",
    "notes",
]


def clean(value: str | None) -> str:
    return "" if value is None else value.strip()


def require_sample_id(value: str, source: str, row_number: int) -> str:
    sample_id = clean(value)
    if len(sample_id) != 5 or not sample_id.isdigit():
        raise ValueError(f"Invalid sample ID {sample_id!r} in {source} row {row_number}")
    return sample_id


def read_jsc(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for number, source in enumerate(csv.DictReader(handle), start=2):
            sample_id = require_sample_id(source.get("sample_id", ""), str(path), number)
            rows.append(
                {
                    "sample_id": sample_id,
                    "material_type": "lunar_soil",
                    "source_dataset": "JSC_19069_Handbook_of_Lunar_Soils",
                    "feo_wt_percent": clean(source.get("feo_wt_percent")),
                    "is_feo": clean(source.get("is_feo")),
                    "al2o3_wt_percent": "",
                    "tio2_wt_percent": "",
                    "mgo_wt_percent": "",
                    "cao_wt_percent": "",
                    "source_viewer_page": clean(source.get("heading_page")),
                    "source_printed_page": "",
                    "table_reference": clean(source.get("heading_text")),
                    "source_url": "",
                    "extraction_status": clean(source.get("status")),
                    "notes": clean(source.get("correction_note")),
                }
            )
    return rows


def read_breccia(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for number, source in enumerate(csv.DictReader(handle), start=2):
            sample_id = require_sample_id(source.get("Sample ID", ""), str(path), number)
            rows.append(
                {
                    "sample_id": sample_id,
                    "material_type": "regolith_breccia",
                    "source_dataset": "LPI_Regolith_Breccia_Workbook",
                    "feo_wt_percent": clean(source.get("FeO (wt.%)")),
                    "is_feo": "",
                    "al2o3_wt_percent": clean(source.get("Al2O3 (wt.%)")),
                    "tio2_wt_percent": clean(source.get("TiO2 (wt.%)")),
                    "mgo_wt_percent": clean(source.get("MgO (wt.%)")),
                    "cao_wt_percent": clean(source.get("CaO (wt.%)")),
                    "source_viewer_page": clean(source.get("PDF viewer page")),
                    "source_printed_page": clean(source.get("Printed page")),
                    "table_reference": clean(source.get("Table reference")),
                    "source_url": clean(source.get("Source URL")),
                    "extraction_status": "visually_checked",
                    "notes": clean(source.get("Notes")),
                }
            )
    return rows


def ensure_unique_within_source(rows: list[dict[str, str]]) -> None:
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["source_dataset"], row["sample_id"])
        if key in seen:
            raise ValueError(f"Duplicate source/sample row: {key}")
        seen.add(key)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsc-audit", type=Path, required=True)
    parser.add_argument("--breccia-composition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    jsc = read_jsc(args.jsc_audit)
    breccia = read_breccia(args.breccia_composition)
    rows = jsc + breccia
    ensure_unique_within_source(rows)
    rows.sort(key=lambda row: (int(row["sample_id"]), row["source_dataset"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "schema": "atlas-source-chemistry-index-1.0",
        "rows": len(rows),
        "distinct_sample_ids": len({row["sample_id"] for row in rows}),
        "jsc_lunar_soil_rows": len(jsc),
        "breccia_rows": len(breccia),
        "rows_with_feo": sum(bool(row["feo_wt_percent"]) for row in rows),
        "rows_with_is_feo": sum(bool(row["is_feo"]) for row in rows),
        "method": "source rows preserved; no averaging, imputation, or parent-child transfer",
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
