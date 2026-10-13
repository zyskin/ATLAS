#!/usr/bin/env python3
"""Mark RELAB spectra suitable or unsuitable for full Adams/M3 comparison.

The comparison range is 461--2500 nm: the first M3 global-band centre is
approximately 460.99 nm, and the Adams spectra extend through the M3 band near
2500 nm.  A spectrum must cover that entire interval.  Long-wavelength-only
FTIR records remain valid RELAB data, but are unsuitable for this comparison.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter
from pathlib import Path


VERSION = "1.0"
REQUIRED_MIN_NM = 461.0
REQUIRED_MAX_NM = 2500.0
OUTPUT_NAME = "relab_m3_suitability_v1.csv"


def clean(value: object) -> str:
    return " ".join(str(value or "").replace("\xa0", " ").split())


def canonical_id(value: object) -> str:
    text = clean(value)
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".", 1)[0]
    return text.upper()


def number(value: object) -> float | None:
    try:
        result = float(clean(value))
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def assess_relab_suitability(
    index_row: dict[str, str],
    required_min_nm: float = REQUIRED_MIN_NM,
    required_max_nm: float = REQUIRED_MAX_NM,
) -> dict[str, object]:
    minimum = number(index_row.get("data_wavelength_min_nm"))
    maximum = number(index_row.get("data_wavelength_max_nm"))
    if minimum is None or maximum is None:
        suitable = False
        coverage_class = "missing_coverage_metadata"
        reason = "missing finite data_wavelength_min_nm or data_wavelength_max_nm"
    else:
        starts_late = minimum > required_min_nm
        ends_early = maximum < required_max_nm
        suitable = not starts_late and not ends_early
        if suitable:
            coverage_class = "full_adams_m3_comparison_range"
            reason = "covers the complete required comparison interval"
        elif minimum >= 1200.0:
            coverage_class = "long_wavelength_only"
            reason = f"starts at {minimum:g} nm, after the required {required_min_nm:g} nm"
        elif starts_late and ends_early:
            coverage_class = "partial_at_both_ends"
            reason = (
                f"covers only {minimum:g}-{maximum:g} nm; required "
                f"{required_min_nm:g}-{required_max_nm:g} nm"
            )
        elif starts_late:
            coverage_class = "starts_after_required_range"
            reason = f"starts at {minimum:g} nm, after the required {required_min_nm:g} nm"
        else:
            coverage_class = "ends_before_required_range"
            reason = f"ends at {maximum:g} nm, before the required {required_max_nm:g} nm"
    return {
        "suitable": suitable,
        "coverage_class": coverage_class,
        "reason": reason,
        "data_wavelength_min_nm": minimum,
        "data_wavelength_max_nm": maximum,
        "required_wavelength_min_nm": required_min_nm,
        "required_wavelength_max_nm": required_max_nm,
    }


def is_standard_geometry(row: dict[str, str], tolerance: float = 0.01) -> bool:
    incidence = number(row.get("incidence_angle_deg"))
    emission = number(row.get("emission_angle_deg"))
    phase = number(row.get("phase_angle_deg"))
    return (
        incidence is not None
        and emission is not None
        and phase is not None
        and abs(incidence - 30.0) <= tolerance
        and abs(emission) <= tolerance
        and abs(phase - 30.0) <= tolerance
    )


def parse_args() -> argparse.Namespace:
    script_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    parser.add_argument("--project-root", type=Path, default=script_root)
    parser.add_argument("--relab-index", type=Path)
    parser.add_argument("--relab-crosswalk", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--required-min-nm", type=float, default=REQUIRED_MIN_NM)
    parser.add_argument("--required-max-nm", type=float, default=REQUIRED_MAX_NM)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.required_min_nm >= args.required_max_nm:
        raise ValueError("--required-min-nm must be smaller than --required-max-nm")
    root = args.project_root.resolve()
    index_path = args.relab_index or (
        root / "relab_raw_native_2026-10-04" / "relab_spectrum_index.csv"
    )
    crosswalk_path = args.relab_crosswalk or (
        root / "crosswalk_output_v2_2026-10-03" / "relab_atlas_crosswalk.csv"
    )
    output_path = args.output or root / "data" / OUTPUT_NAME

    with index_path.open(newline="", encoding="utf-8-sig") as handle:
        index_rows = list(csv.DictReader(handle))
    with crosswalk_path.open(newline="", encoding="utf-8-sig") as handle:
        crosswalk_rows = list(csv.DictReader(handle))
    parents = {
        canonical_id(row.get("measurement_id")): clean(row.get("apollo_parent_sample"))
        for row in crosswalk_rows
    }

    output_rows: list[dict[str, object]] = []
    for row in index_rows:
        measurement = canonical_id(row.get("measurement_id"))
        assessment = assess_relab_suitability(
            row, args.required_min_nm, args.required_max_nm
        )
        output_rows.append(
            {
                "measurement_id": measurement,
                "apollo_parent_id": parents.get(measurement, ""),
                "suitable_for_adams_m3_comparison": (
                    "yes" if assessment["suitable"] else "no"
                ),
                "coverage_class": assessment["coverage_class"],
                "suitability_reason": assessment["reason"],
                "data_wavelength_min_nm": assessment["data_wavelength_min_nm"],
                "data_wavelength_max_nm": assessment["data_wavelength_max_nm"],
                "required_wavelength_min_nm": assessment["required_wavelength_min_nm"],
                "required_wavelength_max_nm": assessment["required_wavelength_max_nm"],
                "standard_30_0_30_geometry": "yes" if is_standard_geometry(row) else "no",
                "instrument_names": clean(row.get("instrument_names")),
                "incidence_angle_deg": clean(row.get("incidence_angle_deg")),
                "emission_angle_deg": clean(row.get("emission_angle_deg")),
                "phase_angle_deg": clean(row.get("phase_angle_deg")),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    classes = Counter(str(row["coverage_class"]) for row in output_rows)
    suitable_count = sum(
        row["suitable_for_adams_m3_comparison"] == "yes" for row in output_rows
    )
    print(f"RELAB measurements assessed: {len(output_rows)}")
    print(f"Suitable for full Adams/M3 comparison: {suitable_count}")
    print(f"Unsuitable: {len(output_rows) - suitable_count}")
    for name, count in sorted(classes.items()):
        print(f"  {name}: {count}")
    print(f"Suitability table: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
