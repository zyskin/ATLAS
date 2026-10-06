#!/usr/bin/env python3
"""Build a preliminary FeO regression table from RELAB spectra in the M3 basis.

The output has one row per unique RELAB measurement, not one row per FeO
claim.  When the comprehensive table contains several rows for the same
measurement, the smallest existing comprehensive-table ``id`` is retained as
the stable ML row id and every contributing id is recorded.

Only measurements with usable M3 bands 1..74 are retained.  FeO claims are
collected from:

* exact RELAB/ancillary chemistry associations in atlas_relab_feo_table.csv;
* automatically accepted bulk Atlas-PDF measurements (needs_review=no);
* accepted rows in review_candidate_adjudications.csv;
* resolved rows in missing_case_resolutions.csv; and
* an optional manually reviewed claims CSV.

Unreviewed PDF candidates and mineral/component-only FeO values are excluded.
The preliminary target is the arithmetic mean of the unique included claims.
All individual values and their provenance remain in the output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


VERSION = "m3-feo-ml-dataset-1.0"
DEFAULT_BANDS = tuple(range(1, 75))
FALSE_WORDS = {"", "0", "false", "no", "n", "none", "null"}


@dataclass(frozen=True)
class Claim:
    value: float
    source_kind: str
    scope: str
    source_id: str
    source_url: str
    reference: str
    note: str


def clean(value: object) -> str:
    return " ".join(str(value or "").split())


def yes(value: object) -> bool:
    return clean(value).casefold() not in FALSE_WORDS


def parse_float(value: object) -> float | None:
    text = clean(value)
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def split_numbers(value: object) -> list[float]:
    return [float(x) for x in re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", clean(value))]


def sample_ids(value: object) -> list[str]:
    # Apollo Atlas identifiers are normally five digits, sometimes followed by
    # a letter.  Do not turn arbitrary chemistry values into sample ids.
    return list(dict.fromkeys(re.findall(r"(?<!\d)(\d{5}[A-Za-z]?)(?!\d)", clean(value))))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def require_columns(path: Path, rows: list[dict[str, str]], required: set[str]) -> None:
    if not rows:
        raise ValueError(f"{path}: no data rows")
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")


def claim_key(claim: Claim) -> tuple[object, ...]:
    # A URL/source-id plus value identifies a repeated extraction of the same
    # claim.  Separate references with the same value remain separate claims.
    locator = clean(claim.source_url).casefold() or clean(claim.source_id).casefold()
    return (round(claim.value, 8), locator, clean(claim.reference).casefold())


def add_parent_claim(store: dict[str, dict[tuple[object, ...], Claim]], parent: str,
                     claim: Claim, warnings: list[str], max_feo: float) -> None:
    if not (0.0 <= claim.value <= max_feo):
        warnings.append(
            f"Excluded implausible FeO={claim.value:g} for parent {parent} "
            f"from {claim.source_kind}:{claim.source_id}"
        )
        return
    store[parent][claim_key(claim)] = claim


def load_automatic_pdf_claims(path: Path | None, store, warnings, max_feo: float) -> None:
    if path is None or not path.exists():
        return
    rows = read_csv(path)
    require_columns(path, rows, {"measurement_id", "atlas_sample_id", "value_wt_percent",
                                "sample_scope", "needs_review"})
    for row in rows:
        if yes(row.get("needs_review")):
            continue
        if clean(row.get("sample_scope")).casefold() != "bulk_or_whole_sample":
            continue
        value = parse_float(row.get("value_wt_percent"))
        if value is None:
            continue
        parents = sample_ids(row.get("atlas_sample_id"))
        if len(parents) != 1:
            warnings.append(
                f"Skipped accepted PDF row {clean(row.get('measurement_id'))}: "
                f"expected one sample id, found {parents or 'none'}"
            )
            continue
        claim = Claim(
            value=value,
            source_kind="atlas_pdf_automatic_accepted",
            scope="bulk_or_whole_sample",
            source_id=f"atlas_pdf_measurement:{clean(row.get('measurement_id'))}",
            source_url=clean(row.get("source_url")),
            reference=clean(row.get("reference")),
            note=f"PDF page {clean(row.get('pdf_page'))}; {clean(row.get('evidence_text'))}",
        )
        add_parent_claim(store, parents[0], claim, warnings, max_feo)


def load_adjudicated_claims(path: Path | None, store, warnings, max_feo: float) -> None:
    if path is None or not path.exists():
        return
    rows = read_csv(path)
    require_columns(path, rows, {"measurement_id", "decision", "assigned_sample_id",
                                "value_wt_percent"})
    for row in rows:
        if clean(row.get("decision")) != "accepted_parent_or_documented_fraction":
            continue
        value = parse_float(row.get("value_wt_percent"))
        parents = sample_ids(row.get("assigned_sample_id"))
        if value is None or len(parents) != 1:
            warnings.append(
                f"Skipped adjudication {clean(row.get('measurement_id'))}: "
                f"value={row.get('value_wt_percent')!r}, sample ids={parents}"
            )
            continue
        claim = Claim(
            value=value,
            source_kind="atlas_pdf_ai_adjudicated",
            scope=clean(row.get("sample_scope")) or "parent_or_documented_fraction",
            source_id=f"atlas_pdf_measurement:{clean(row.get('measurement_id'))}",
            source_url=clean(row.get("source_url")),
            reference=clean(row.get("reference")),
            note=clean(row.get("adjudication_reason")),
        )
        add_parent_claim(store, parents[0], claim, warnings, max_feo)


def load_missing_case_claims(path: Path | None, store, warnings, max_feo: float) -> None:
    if path is None or not path.exists():
        return
    rows = read_csv(path)
    require_columns(path, rows, {"queue_id", "apollo_parent_id", "resolution_status",
                                "accepted_feo_wt_percent"})
    for row in rows:
        if not clean(row.get("resolution_status")).startswith("resolved_"):
            continue
        parents = sample_ids(row.get("apollo_parent_id"))
        if len(parents) != 1:
            continue
        values = split_numbers(row.get("accepted_feo_wt_percent"))
        for index, value in enumerate(values, 1):
            claim = Claim(
                value=value,
                source_kind="publication_ai_resolved",
                scope=clean(row.get("accepted_scope")) or "resolved_sample_or_fraction",
                source_id=f"missing_case:{clean(row.get('queue_id'))}:{index}",
                source_url=clean(row.get("primary_source_url")),
                reference=clean(row.get("source_locator")),
                note=clean(row.get("review_note")) or clean(row.get("evidence_summary")),
            )
            add_parent_claim(store, parents[0], claim, warnings, max_feo)


def load_manual_claims(path: Path | None, store, warnings, max_feo: float) -> None:
    if path is None or not path.exists():
        return
    rows = read_csv(path)
    require_columns(path, rows, {"apollo_parent_id", "value_wt_percent", "include"})
    for row_number, row in enumerate(rows, 2):
        if not yes(row.get("include")):
            continue
        parents = sample_ids(row.get("apollo_parent_id"))
        value = parse_float(row.get("value_wt_percent"))
        if len(parents) != 1 or value is None:
            warnings.append(f"Skipped invalid manual claim at {path}:{row_number}")
            continue
        claim = Claim(
            value=value,
            source_kind=clean(row.get("source_kind")) or "manual_reviewed",
            scope=clean(row.get("scope")) or "bulk_or_whole_sample",
            source_id=clean(row.get("source_id")) or f"manual:{row_number}",
            source_url=clean(row.get("source_url")),
            reference=clean(row.get("reference")),
            note=clean(row.get("note")),
        )
        add_parent_claim(store, parents[0], claim, warnings, max_feo)


def iron_angle_principal(r750: float, r950: float, x0: float, y0: float) -> float:
    # Eq. (4) of Ajith Kumar & Kumar (2014) is written with the principal
    # one-argument inverse tangent, rather than an atan2 quadrant convention.
    denominator = r750 - x0
    if abs(denominator) < 1e-15:
        return math.copysign(math.pi / 2.0, -(r950 / r750 - y0))
    return -math.atan((r950 / r750 - y0) / denominator)


def baseline_values(bands: dict[int, float]) -> dict[str, object]:
    r750 = bands[9]
    r950 = bands[19]
    theta = iron_angle_principal(r750, r950, 0.08, 1.19)

    mean750 = statistics.fmean(bands[b] for b in (8, 9, 10))
    mean950 = statistics.fmean(bands[b] for b in (18, 19, 20))
    h_fe = iron_angle_principal(mean750, mean950, 0.08, 1.18)
    return {
        "r750_direct_b09": r750,
        "r950_direct_b19": r950,
        "r950_over_r750_direct": r950 / r750,
        "lucey_direct_h_fe": theta,
        "lucey_direct_feo_wt_percent": 17.427 * theta - 7.565,
        "r750_mean_b08_b10": mean750,
        "r950_mean_b18_b20": mean950,
        "r950_over_r750_mean": mean950 / mean750,
        "kumar_m3_h_fe": h_fe,
        "kumar_m3_linear_feo_wt_percent": 12.92 * h_fe - 2.107,
        "kumar_m3_power_feo_wt_percent": 10.59 * (h_fe ** 1.209) if h_fe >= 0 else "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comprehensive-table", required=True, type=Path)
    parser.add_argument("--m3-spectra", required=True, type=Path)
    parser.add_argument("--atlas-pdf-feo", type=Path)
    parser.add_argument("--adjudications", type=Path)
    parser.add_argument("--missing-resolutions", type=Path)
    parser.add_argument("--manual-feo", type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument("--max-feo", type=float, default=35.0)
    args = parser.parse_args()

    warnings: list[str] = []
    comprehensive = read_csv(args.comprehensive_table)
    require_columns(
        args.comprehensive_table,
        comprehensive,
        {"id", "i_atlas_sample", "j_child", "k_feo_wt_percent", "feo_known",
         "relab_measurement_id", "relab_specimen_id", "relab_specimen_name"},
    )

    by_measurement: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in comprehensive:
        measurement = clean(row.get("relab_measurement_id"))
        if measurement:
            by_measurement[measurement].append(row)

    parent_claims: dict[str, dict[tuple[object, ...], Claim]] = defaultdict(dict)
    load_automatic_pdf_claims(args.atlas_pdf_feo, parent_claims, warnings, args.max_feo)
    load_adjudicated_claims(args.adjudications, parent_claims, warnings, args.max_feo)
    load_missing_case_claims(args.missing_resolutions, parent_claims, warnings, args.max_feo)
    load_manual_claims(args.manual_feo, parent_claims, warnings, args.max_feo)

    spectra: dict[str, dict[int, float]] = defaultdict(dict)
    centroids: dict[int, float] = {}
    with args.m3_spectra.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"measurement_id", "m3_band", "m3_response_centroid_nm",
                    "m3_reflectance", "usable"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{args.m3_spectra}: missing columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, 2):
            measurement = clean(row.get("measurement_id"))
            if measurement not in by_measurement:
                continue
            try:
                band = int(clean(row.get("m3_band")))
            except ValueError:
                continue
            if band not in DEFAULT_BANDS or not yes(row.get("usable")):
                continue
            value = parse_float(row.get("m3_reflectance"))
            centroid = parse_float(row.get("m3_response_centroid_nm"))
            if value is None or centroid is None:
                continue
            if band in spectra[measurement]:
                raise ValueError(
                    f"{args.m3_spectra}:{row_number}: duplicate band {band} for {measurement}"
                )
            spectra[measurement][band] = value
            if band in centroids and abs(centroids[band] - centroid) > 1e-6:
                raise ValueError(f"Inconsistent M3 centroid for band {band}")
            centroids[band] = centroid

    rows_out: list[dict[str, object]] = []
    excluded_no_74 = 0
    excluded_no_feo = 0

    for measurement, rows in by_measurement.items():
        band_values = spectra.get(measurement, {})
        if set(band_values) != set(DEFAULT_BANDS):
            excluded_no_74 += 1
            continue

        parent_values = {clean(row.get("i_atlas_sample")) for row in rows}
        child_values = {clean(row.get("j_child")) for row in rows}
        specimen_values = {clean(row.get("relab_specimen_id")) for row in rows}
        if len(parent_values) != 1 or len(child_values) != 1 or len(specimen_values) != 1:
            warnings.append(f"Excluded {measurement}: inconsistent comprehensive-table identity")
            continue
        parent = next(iter(parent_values))

        exact: dict[tuple[object, ...], Claim] = {}
        for row in rows:
            value = parse_float(row.get("k_feo_wt_percent"))
            if not yes(row.get("feo_known")) or value is None or value < 0:
                continue
            claim = Claim(
                value=value,
                source_kind="relab_ancillary_chemistry",
                scope=clean(row.get("chemistry_phase")) or "unspecified",
                source_id=clean(row.get("chemistry_product_id")) or clean(row.get("chemistry_specimen_id")),
                source_url="",
                reference=clean(row.get("chemistry_source_and_method")),
                note="Exact association recorded in atlas_relab_feo_table.csv",
            )
            if 0 <= value <= args.max_feo:
                exact[claim_key(claim)] = claim
            else:
                warnings.append(f"Excluded implausible direct FeO={value:g} for {measurement}")

        combined: dict[tuple[object, ...], Claim] = {}
        for claim in list(exact.values()) + list(parent_claims.get(parent, {}).values()):
            combined[claim_key(claim)] = claim
        claims = sorted(combined.values(), key=lambda c: (c.value, c.source_kind, c.source_id))
        if not claims:
            excluded_no_feo += 1
            continue

        ids = sorted({int(clean(row["id"])) for row in rows})
        first = min(rows, key=lambda row: int(clean(row["id"])))
        values = [claim.value for claim in claims]
        out: dict[str, object] = {
            "id": min(ids),
            "comprehensive_table_ids": ";".join(map(str, ids)),
            "i_atlas_sample": parent,
            "j_child": next(iter(child_values)),
            "relab_measurement_id": measurement,
            "relab_specimen_id": next(iter(specimen_values)),
            "relab_specimen_name": clean(first.get("relab_specimen_name")),
            "atlas_sample_type": clean(first.get("atlas_sample_type")),
            "atlas_lithology": clean(first.get("atlas_lithology")),
            "crosswalk_needs_review": clean(first.get("crosswalk_needs_review")),
            "feo_mean_wt_percent": statistics.fmean(values),
            "feo_claim_count": len(values),
            "feo_direct_claim_count": len(exact),
            "feo_parent_claim_count": len(claims) - len(exact),
            "feo_min_wt_percent": min(values),
            "feo_max_wt_percent": max(values),
            "feo_sample_std_wt_percent": statistics.stdev(values) if len(values) > 1 else "",
            "feo_values_wt_percent": ";".join(format(v, ".10g") for v in values),
            "feo_claims_json": json.dumps([asdict(c) for c in claims], ensure_ascii=False,
                                           separators=(",", ":")),
        }
        out.update(baseline_values(band_values))
        for band in DEFAULT_BANDS:
            out[f"m3_b{band:02d}"] = band_values[band]
        rows_out.append(out)

    rows_out.sort(key=lambda row: int(row["id"]))
    if not rows_out:
        raise RuntimeError("No rows satisfied both the 74-band and FeO requirements")

    metadata_columns = [
        "id", "comprehensive_table_ids", "i_atlas_sample", "j_child",
        "relab_measurement_id", "relab_specimen_id", "relab_specimen_name",
        "atlas_sample_type", "atlas_lithology", "crosswalk_needs_review",
        "feo_mean_wt_percent", "feo_claim_count", "feo_direct_claim_count",
        "feo_parent_claim_count", "feo_min_wt_percent", "feo_max_wt_percent",
        "feo_sample_std_wt_percent", "feo_values_wt_percent", "feo_claims_json",
        "r750_direct_b09", "r950_direct_b19", "r950_over_r750_direct",
        "lucey_direct_h_fe", "lucey_direct_feo_wt_percent",
        "r750_mean_b08_b10", "r950_mean_b18_b20", "r950_over_r750_mean",
        "kumar_m3_h_fe", "kumar_m3_linear_feo_wt_percent",
        "kumar_m3_power_feo_wt_percent",
    ]
    fieldnames = metadata_columns + [f"m3_b{band:02d}" for band in DEFAULT_BANDS]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    inputs = {}
    for label, path in {
        "comprehensive_table": args.comprehensive_table,
        "m3_spectra": args.m3_spectra,
        "atlas_pdf_feo": args.atlas_pdf_feo,
        "adjudications": args.adjudications,
        "missing_resolutions": args.missing_resolutions,
        "manual_feo": args.manual_feo,
    }.items():
        if path is not None and path.exists():
            inputs[label] = {"path": str(path.resolve()), "sha256": sha256(path)}

    manifest = {
        "schema": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs,
        "output_csv": str(args.output_csv.resolve()),
        "row_count": len(rows_out),
        "unique_apollo_parent_count": len({row["i_atlas_sample"] for row in rows_out}),
        "required_m3_bands": list(DEFAULT_BANDS),
        "m3_band_centroids_nm": {str(b): centroids.get(b) for b in DEFAULT_BANDS},
        "feo_target": "arithmetic mean of unique accepted/direct associated claims",
        "excluded_measurements_without_all_74_bands": excluded_no_74,
        "excluded_measurements_without_accepted_feo": excluded_no_feo,
        "warnings": warnings,
        "important_notes": [
            "One ML row represents one unique RELAB measurement.",
            "The row id is the smallest existing comprehensive-table id for that measurement; it is not renumbered.",
            "All replicates and rows from the same Apollo parent must remain in the same validation fold.",
            "Unreviewed PDF candidates and component/mineral-only FeO values are excluded.",
            "Parent-level FeO claims are provisionally associated with spectra of that Apollo parent for this preliminary experiment.",
        ],
    }
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"ML rows written: {len(rows_out)}")
    print(f"Unique Apollo parents: {manifest['unique_apollo_parent_count']}")
    print(f"Excluded without all 74 usable bands: {excluded_no_74}")
    print(f"Excluded without accepted FeO: {excluded_no_feo}")
    print(f"Dataset: {args.output_csv}")
    print(f"Manifest: {args.manifest_json}")
    if warnings:
        print(f"Warnings: {len(warnings)} (recorded in manifest)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
