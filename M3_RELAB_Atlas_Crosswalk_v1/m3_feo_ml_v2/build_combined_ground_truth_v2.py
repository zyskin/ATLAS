#!/usr/bin/env python3
"""Build the combined FeO/Is-FeO ground-truth table in the M3 spectral basis.

One output row represents one RELAB measurement with all usable M3 bands 1--74.
Chemical and maturity claims stay source-resolved in JSON columns.  The numeric
targets are arithmetic means of unique accepted claims, for this experiment
only; no source rows are destroyed or overwritten.

The program distinguishes chemistry explicitly linked to a RELAB measurement
from values linked only to the Apollo parent sample.  Parent-level values are
provisionally attached to every spectrum cross-walked to that parent, and that
association is recorded in ``feo_target_scope``.
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
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


VERSION = "m3-feo-isfeo-ground-truth-2.0"
BANDS = tuple(range(1, 75))
FALSE_WORDS = {"", "0", "false", "no", "n", "none", "null"}

ORIGIN_CHANGED_X0 = 0.03433418510481713
ORIGIN_CHANGED_Y0 = 1.2571662682003644
ORIGIN_CHANGED_A = 6.274305964663926
ORIGIN_CHANGED_B = 9.740661344354981


@dataclass(frozen=True)
class Claim:
    value: float
    source_kind: str
    scope: str
    association: str
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
    return list(dict.fromkeys(re.findall(r"(?<!\d)(\d{5}[A-Za-z]?)(?!\d)", clean(value))))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def require_columns(path: Path, rows: list[dict[str, str]], required: set[str]) -> None:
    if not rows:
        raise ValueError(f"{path}: no data rows")
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def claim_key(claim: Claim) -> tuple[object, ...]:
    locator = clean(claim.source_url).casefold() or clean(claim.source_id).casefold()
    return (round(claim.value, 8), locator, clean(claim.reference).casefold())


def add_parent_claim(store, parent: str, claim: Claim, warnings: list[str],
                     minimum: float, maximum: float) -> None:
    if not (minimum <= claim.value <= maximum):
        warnings.append(
            f"Excluded {claim.source_kind} value {claim.value:g} for {parent}; "
            f"accepted interval is [{minimum:g}, {maximum:g}]"
        )
        return
    store[parent][claim_key(claim)] = claim


def load_automatic_pdf_feo(path: Path | None, store, warnings, max_feo: float) -> int:
    if path is None or not path.exists():
        return 0
    rows = read_csv(path)
    require_columns(path, rows, {"measurement_id", "atlas_sample_id", "value_wt_percent",
                                "sample_scope", "needs_review"})
    count = 0
    for row in rows:
        if yes(row.get("needs_review")):
            continue
        if clean(row.get("sample_scope")).casefold() != "bulk_or_whole_sample":
            continue
        value = parse_float(row.get("value_wt_percent"))
        parents = sample_ids(row.get("atlas_sample_id"))
        if value is None or len(parents) != 1:
            continue
        claim = Claim(
            value=value,
            source_kind="atlas_pdf_automatic_accepted",
            scope="bulk_or_whole_sample",
            association="apollo_parent",
            source_id=f"atlas_pdf_measurement:{clean(row.get('measurement_id'))}",
            source_url=clean(row.get("source_url")),
            reference=clean(row.get("reference")),
            note=f"PDF page {clean(row.get('pdf_page'))}; {clean(row.get('evidence_text'))}",
        )
        add_parent_claim(store, parents[0], claim, warnings, 0.01, max_feo)
        count += 1
    return count


def load_adjudicated_feo(path: Path | None, store, warnings, max_feo: float) -> int:
    if path is None or not path.exists():
        return 0
    rows = read_csv(path)
    require_columns(path, rows, {"measurement_id", "decision", "assigned_sample_id",
                                "value_wt_percent"})
    count = 0
    for row in rows:
        if clean(row.get("decision")) != "accepted_parent_or_documented_fraction":
            continue
        value = parse_float(row.get("value_wt_percent"))
        parents = sample_ids(row.get("assigned_sample_id"))
        if value is None or len(parents) != 1:
            continue
        claim = Claim(
            value=value,
            source_kind="atlas_pdf_ai_adjudicated",
            scope=clean(row.get("sample_scope")) or "parent_or_documented_fraction",
            association="apollo_parent",
            source_id=f"atlas_pdf_measurement:{clean(row.get('measurement_id'))}",
            source_url=clean(row.get("source_url")),
            reference=clean(row.get("reference")),
            note=clean(row.get("adjudication_reason")),
        )
        add_parent_claim(store, parents[0], claim, warnings, 0.01, max_feo)
        count += 1
    return count


def load_resolved_feo(path: Path | None, store, warnings, max_feo: float) -> int:
    if path is None or not path.exists():
        return 0
    rows = read_csv(path)
    require_columns(path, rows, {"queue_id", "apollo_parent_id", "resolution_status",
                                "accepted_feo_wt_percent"})
    count = 0
    for row in rows:
        if not clean(row.get("resolution_status")).startswith("resolved_"):
            continue
        parents = sample_ids(row.get("apollo_parent_id"))
        if len(parents) != 1:
            continue
        for index, value in enumerate(split_numbers(row.get("accepted_feo_wt_percent")), 1):
            claim = Claim(
                value=value,
                source_kind="publication_ai_resolved",
                scope=clean(row.get("accepted_scope")) or "resolved_sample_or_fraction",
                association="apollo_parent",
                source_id=f"missing_case:{clean(row.get('queue_id'))}:{index}",
                source_url=clean(row.get("primary_source_url")),
                reference=clean(row.get("source_locator")),
                note=clean(row.get("review_note")) or clean(row.get("evidence_summary")),
            )
            add_parent_claim(store, parents[0], claim, warnings, 0.01, max_feo)
            count += 1
    return count


def load_manual_feo(path: Path | None, store, warnings, max_feo: float) -> int:
    if path is None or not path.exists():
        return 0
    rows = read_csv(path)
    require_columns(path, rows, {"apollo_parent_id", "value_wt_percent", "include"})
    count = 0
    for row_number, row in enumerate(rows, 2):
        if not yes(row.get("include")):
            continue
        parents = sample_ids(row.get("apollo_parent_id"))
        value = parse_float(row.get("value_wt_percent"))
        if len(parents) != 1 or value is None:
            warnings.append(f"Skipped invalid manual FeO claim at {path}:{row_number}")
            continue
        claim = Claim(
            value=value,
            source_kind=clean(row.get("source_kind")) or "manual_reviewed",
            scope=clean(row.get("scope")) or "bulk_or_whole_sample",
            association="apollo_parent",
            source_id=clean(row.get("source_id")) or f"manual:{row_number}",
            source_url=clean(row.get("source_url")),
            reference=clean(row.get("reference")),
            note=clean(row.get("note")),
        )
        add_parent_claim(store, parents[0], claim, warnings, 0.01, max_feo)
        count += 1
    return count


def load_source_chemistry(path: Path | None, feo_store, is_store, warnings,
                          max_feo: float, max_is_feo: float) -> dict[str, int]:
    counts = Counter()
    if path is None or not path.exists():
        return dict(counts)
    rows = read_csv(path)
    require_columns(path, rows, {"sample_id", "source_dataset", "feo_wt_percent",
                                "is_feo", "source_url", "table_reference"})
    for row_number, row in enumerate(rows, 2):
        parents = sample_ids(row.get("sample_id"))
        if len(parents) != 1:
            warnings.append(f"Skipped source-chemistry row {row_number}: invalid sample id")
            continue
        parent = parents[0]
        dataset = clean(row.get("source_dataset"))
        scope = clean(row.get("material_type")) or "apollo_sample"
        common = {
            "scope": scope,
            "association": "apollo_parent",
            "source_id": f"{dataset}:{parent}",
            "source_url": clean(row.get("source_url")),
            "reference": clean(row.get("table_reference")),
            "note": clean(row.get("notes")),
        }
        feo = parse_float(row.get("feo_wt_percent"))
        if feo is not None:
            add_parent_claim(
                feo_store, parent,
                Claim(value=feo, source_kind=f"source_extract:{dataset}", **common),
                warnings, 0.01, max_feo,
            )
            counts["feo_rows"] += 1
        maturity = parse_float(row.get("is_feo"))
        if maturity is not None:
            add_parent_claim(
                is_store, parent,
                Claim(value=maturity, source_kind=f"source_extract:{dataset}", **common),
                warnings, 0.0, max_is_feo,
            )
            counts["is_feo_rows"] += 1
    return dict(counts)


def load_existing_is_feo(path: Path | None, store, warnings, max_is_feo: float) -> dict[str, int]:
    counts = Counter()
    if path is None or not path.exists():
        return dict(counts)
    rows = read_csv(path)
    require_columns(path, rows, {"sample_id", "value_text", "needs_review", "source_url"})
    for row_number, row in enumerate(rows, 2):
        if clean(row.get("needs_review")).casefold() != "no":
            counts["excluded_not_confirmed"] += 1
            continue
        parents = sample_ids(row.get("sample_id"))
        value = parse_float(row.get("value_text"))
        if len(parents) != 1 or value is None:
            counts["excluded_invalid"] += 1
            continue
        claim = Claim(
            value=value,
            source_kind="existing_reviewed_is_feo",
            scope=clean(row.get("scope")) or "sample",
            association="apollo_parent",
            source_id=f"existing_is_feo:{row_number}",
            source_url=clean(row.get("source_url")),
            reference=clean(row.get("source_locator")),
            note=clean(row.get("evidence")),
        )
        add_parent_claim(store, parents[0], claim, warnings, 0.0, max_is_feo)
        counts["accepted"] += 1
    return dict(counts)


def principal_angle(x: float, y: float, x0: float, y0: float) -> float:
    denominator = x - x0
    numerator = y - y0
    if abs(denominator) < 1e-15:
        return math.copysign(math.pi / 2.0, -numerator)
    return -math.atan(numerator / denominator)


def angle_2pi(x: float, y: float, x0: float, y0: float) -> float:
    return (-math.atan2(y - y0, x - x0)) % (2.0 * math.pi)


def baseline_values(bands: dict[int, float]) -> dict[str, object]:
    r750 = bands[9]
    r950 = bands[19]
    ratio = r950 / r750
    h_standard = principal_angle(r750, ratio, 0.08, 1.19)
    h_standard_2pi = angle_2pi(r750, ratio, 0.08, 1.19)
    h_changed = principal_angle(r750, ratio, ORIGIN_CHANGED_X0, ORIGIN_CHANGED_Y0)
    h_changed_2pi = angle_2pi(r750, ratio, ORIGIN_CHANGED_X0, ORIGIN_CHANGED_Y0)

    mean750 = statistics.fmean(bands[b] for b in (8, 9, 10))
    mean950 = statistics.fmean(bands[b] for b in (18, 19, 20))
    mean_ratio = mean950 / mean750
    h_kumar = principal_angle(mean750, mean_ratio, 0.08, 1.18)
    h_kumar_2pi = angle_2pi(mean750, mean_ratio, 0.08, 1.18)
    return {
        "r750_direct_b09": r750,
        "r950_direct_b19": r950,
        "r950_over_r750_direct": ratio,
        "lucey_standard_h_principal": h_standard,
        "lucey_standard_feo_principal": 17.427 * h_standard - 7.565,
        "lucey_standard_h_atan2_2pi": h_standard_2pi,
        "lucey_standard_feo_atan2_2pi": 17.427 * h_standard_2pi - 7.565,
        "lucey_origin_changed_h_principal": h_changed,
        "lucey_origin_changed_feo_principal": ORIGIN_CHANGED_A * h_changed + ORIGIN_CHANGED_B,
        "lucey_origin_changed_h_atan2_2pi": h_changed_2pi,
        "lucey_origin_changed_feo_atan2_2pi": ORIGIN_CHANGED_A * h_changed_2pi + ORIGIN_CHANGED_B,
        "r750_mean_b08_b10": mean750,
        "r950_mean_b18_b20": mean950,
        "r950_over_r750_mean": mean_ratio,
        "kumar_h_principal": h_kumar,
        "kumar_feo_linear_principal": 12.92 * h_kumar - 2.107,
        "kumar_feo_power_principal": 10.59 * h_kumar ** 1.209 if h_kumar >= 0 else "",
        "kumar_h_atan2_2pi": h_kumar_2pi,
        "kumar_feo_linear_atan2_2pi": 12.92 * h_kumar_2pi - 2.107,
        "kumar_feo_power_atan2_2pi": 10.59 * h_kumar_2pi ** 1.209,
    }


def summarize_claims(claims: list[Claim], prefix: str) -> dict[str, object]:
    values = [claim.value for claim in claims]
    if not values:
        return {
            f"{prefix}_mean": "", f"{prefix}_claim_count": 0,
            f"{prefix}_min": "", f"{prefix}_max": "", f"{prefix}_sample_std": "",
            f"{prefix}_values": "", f"{prefix}_claims_json": "[]",
        }
    return {
        f"{prefix}_mean": statistics.fmean(values),
        f"{prefix}_claim_count": len(values),
        f"{prefix}_min": min(values),
        f"{prefix}_max": max(values),
        f"{prefix}_sample_std": statistics.stdev(values) if len(values) > 1 else "",
        f"{prefix}_values": ";".join(format(value, ".10g") for value in values),
        f"{prefix}_claims_json": json.dumps(
            [asdict(claim) for claim in claims], ensure_ascii=False, separators=(",", ":")
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comprehensive-table", required=True, type=Path)
    parser.add_argument("--m3-spectra", required=True, type=Path)
    parser.add_argument("--atlas-pdf-feo", type=Path)
    parser.add_argument("--adjudications", type=Path)
    parser.add_argument("--missing-resolutions", type=Path)
    parser.add_argument("--manual-feo", type=Path)
    parser.add_argument("--source-chemistry-index", type=Path)
    parser.add_argument("--existing-is-feo", type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument("--claims-csv", type=Path)
    parser.add_argument("--parent-summary-csv", type=Path)
    parser.add_argument("--max-feo", type=float, default=35.0)
    parser.add_argument("--max-is-feo", type=float, default=200.0)
    args = parser.parse_args()

    warnings: list[str] = []
    comprehensive = read_csv(args.comprehensive_table)
    require_columns(
        args.comprehensive_table, comprehensive,
        {"id", "i_atlas_sample", "j_child", "k_feo_wt_percent", "feo_known",
         "relab_measurement_id", "relab_specimen_id", "relab_specimen_name"},
    )
    by_measurement: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in comprehensive:
        measurement = clean(row.get("relab_measurement_id"))
        if measurement:
            by_measurement[measurement].append(row)

    feo_parent: dict[str, dict[tuple[object, ...], Claim]] = defaultdict(dict)
    is_parent: dict[str, dict[tuple[object, ...], Claim]] = defaultdict(dict)
    input_counts = {
        "automatic_pdf_feo": load_automatic_pdf_feo(
            args.atlas_pdf_feo, feo_parent, warnings, args.max_feo),
        "adjudicated_feo": load_adjudicated_feo(
            args.adjudications, feo_parent, warnings, args.max_feo),
        "resolved_feo": load_resolved_feo(
            args.missing_resolutions, feo_parent, warnings, args.max_feo),
        "manual_feo": load_manual_feo(
            args.manual_feo, feo_parent, warnings, args.max_feo),
    }
    input_counts["source_chemistry"] = load_source_chemistry(
        args.source_chemistry_index, feo_parent, is_parent, warnings,
        args.max_feo, args.max_is_feo,
    )
    input_counts["existing_is_feo"] = load_existing_is_feo(
        args.existing_is_feo, is_parent, warnings, args.max_is_feo,
    )

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
            if band not in BANDS or not yes(row.get("usable")):
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
    excluded_no_target = 0
    identity_warnings = 0
    for measurement, source_rows in by_measurement.items():
        band_values = spectra.get(measurement, {})
        if set(band_values) != set(BANDS):
            excluded_no_74 += 1
            continue

        parents = {clean(row.get("i_atlas_sample")) for row in source_rows}
        children = {clean(row.get("j_child")) for row in source_rows}
        specimens = {clean(row.get("relab_specimen_id")) for row in source_rows}
        if len(parents) != 1 or len(children) != 1 or len(specimens) != 1:
            warnings.append(f"Excluded {measurement}: inconsistent comprehensive identity")
            identity_warnings += 1
            continue
        parent = next(iter(parents))

        exact: dict[tuple[object, ...], Claim] = {}
        for row in source_rows:
            value = parse_float(row.get("k_feo_wt_percent"))
            if not yes(row.get("feo_known")) or value is None or value < 0:
                continue
            claim = Claim(
                value=value,
                source_kind="relab_ancillary_chemistry",
                scope=clean(row.get("chemistry_phase")) or "unspecified",
                association="exact_relab_measurement",
                source_id=clean(row.get("chemistry_product_id"))
                or clean(row.get("chemistry_specimen_id")),
                source_url="",
                reference=clean(row.get("chemistry_source_and_method")),
                note="Exact association recorded in atlas_relab_feo_table.csv",
            )
            if 0.01 <= value <= args.max_feo:
                exact[claim_key(claim)] = claim
            elif value == 0.0:
                warnings.append(
                    f"Excluded zero FeO placeholder for {measurement} from exact chemistry"
                )

        combined = dict(exact)
        combined.update(feo_parent.get(parent, {}))
        feo_claims = sorted(
            combined.values(), key=lambda claim: (claim.value, claim.source_kind, claim.source_id)
        )
        maturity_claims = sorted(
            is_parent.get(parent, {}).values(),
            key=lambda claim: (claim.value, claim.source_kind, claim.source_id),
        )
        if not feo_claims and not maturity_claims:
            excluded_no_target += 1
            continue

        ids = sorted({int(clean(row["id"])) for row in source_rows})
        first = min(source_rows, key=lambda row: int(clean(row["id"])))
        parent_count = len(feo_claims) - len(exact)
        if not feo_claims:
            target_scope = "is_feo_only"
        elif exact and parent_count:
            target_scope = "mixed_exact_and_parent"
        elif exact:
            target_scope = "exact_relab_association_only"
        else:
            target_scope = "parent_level_only"
        out: dict[str, object] = {
            "id": min(ids),
            "comprehensive_table_ids": ";".join(map(str, ids)),
            "i_atlas_sample": parent,
            "j_child": next(iter(children)),
            "relab_measurement_id": measurement,
            "relab_specimen_id": next(iter(specimens)),
            "relab_specimen_name": clean(first.get("relab_specimen_name")),
            "atlas_sample_type": clean(first.get("atlas_sample_type")),
            "atlas_lithology": clean(first.get("atlas_lithology")),
            "crosswalk_needs_review": clean(first.get("crosswalk_needs_review")),
            "feo_target_scope": target_scope,
            "feo_direct_claim_count": len(exact),
            "feo_parent_claim_count": parent_count,
        }
        direct_claims = sorted(
            exact.values(), key=lambda claim: (claim.value, claim.source_kind, claim.source_id)
        )
        parent_claims = sorted(
            feo_parent.get(parent, {}).values(),
            key=lambda claim: (claim.value, claim.source_kind, claim.source_id),
        )
        out.update(summarize_claims(feo_claims, "feo_wt_percent"))
        out.update(summarize_claims(direct_claims, "feo_direct_wt_percent"))
        out.update(summarize_claims(parent_claims, "feo_parent_wt_percent"))
        out.update(summarize_claims(maturity_claims, "is_feo"))
        out.update(baseline_values(band_values))
        for band in BANDS:
            out[f"m3_b{band:02d}"] = band_values[band]
        rows_out.append(out)

    rows_out.sort(key=lambda row: int(row["id"]))
    if not rows_out:
        raise RuntimeError("No rows satisfied both the 74-band and FeO requirements")

    metadata = [
        "id", "comprehensive_table_ids", "i_atlas_sample", "j_child",
        "relab_measurement_id", "relab_specimen_id", "relab_specimen_name",
        "atlas_sample_type", "atlas_lithology", "crosswalk_needs_review",
        "feo_target_scope", "feo_direct_claim_count", "feo_parent_claim_count",
        "feo_wt_percent_mean", "feo_wt_percent_claim_count", "feo_wt_percent_min",
        "feo_wt_percent_max", "feo_wt_percent_sample_std", "feo_wt_percent_values",
        "feo_wt_percent_claims_json",
        "feo_direct_wt_percent_mean", "feo_direct_wt_percent_claim_count",
        "feo_direct_wt_percent_min", "feo_direct_wt_percent_max",
        "feo_direct_wt_percent_sample_std", "feo_direct_wt_percent_values",
        "feo_direct_wt_percent_claims_json",
        "feo_parent_wt_percent_mean", "feo_parent_wt_percent_claim_count",
        "feo_parent_wt_percent_min", "feo_parent_wt_percent_max",
        "feo_parent_wt_percent_sample_std", "feo_parent_wt_percent_values",
        "feo_parent_wt_percent_claims_json",
        "is_feo_mean", "is_feo_claim_count", "is_feo_min", "is_feo_max",
        "is_feo_sample_std", "is_feo_values", "is_feo_claims_json",
        "r750_direct_b09", "r950_direct_b19", "r950_over_r750_direct",
        "lucey_standard_h_principal", "lucey_standard_feo_principal",
        "lucey_standard_h_atan2_2pi", "lucey_standard_feo_atan2_2pi",
        "lucey_origin_changed_h_principal", "lucey_origin_changed_feo_principal",
        "lucey_origin_changed_h_atan2_2pi", "lucey_origin_changed_feo_atan2_2pi",
        "r750_mean_b08_b10", "r950_mean_b18_b20", "r950_over_r750_mean",
        "kumar_h_principal", "kumar_feo_linear_principal",
        "kumar_feo_power_principal", "kumar_h_atan2_2pi",
        "kumar_feo_linear_atan2_2pi", "kumar_feo_power_atan2_2pi",
    ]
    fieldnames = metadata + [f"m3_b{band:02d}" for band in BANDS]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    claims_path = args.claims_csv or args.output_csv.with_name(
        "m3_feo_isfeo_claims_v2.csv"
    )
    parent_path = args.parent_summary_csv or args.output_csv.with_name(
        "m3_feo_isfeo_parent_summary_v2.csv"
    )
    claim_links: dict[tuple[object, ...], dict[str, object]] = {}
    parent_claim_keys: dict[str, dict[str, set[tuple[object, ...]]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for row in rows_out:
        parent = str(row["i_atlas_sample"])
        measurement = str(row["relab_measurement_id"])
        for variable, column in (
            ("FeO_wt_percent", "feo_wt_percent_claims_json"),
            ("Is_FeO", "is_feo_claims_json"),
        ):
            for raw_claim in json.loads(str(row[column])):
                key = (
                    variable, parent, round(float(raw_claim["value"]), 8),
                    raw_claim["source_kind"], raw_claim["association"],
                    raw_claim["source_id"], raw_claim["source_url"],
                    raw_claim["reference"],
                )
                if key not in claim_links:
                    claim_links[key] = {
                        "variable": variable,
                        "i_atlas_sample": parent,
                        **raw_claim,
                        "measurement_ids": set(),
                        "stable_ids": set(),
                    }
                claim_links[key]["measurement_ids"].add(measurement)
                claim_links[key]["stable_ids"].add(str(row["id"]))
                parent_claim_keys[parent][variable].add(key)

    claims_rows = []
    for record in claim_links.values():
        measurements = sorted(record.pop("measurement_ids"))
        stable_ids = sorted(record.pop("stable_ids"), key=int)
        claims_rows.append({
            **record,
            "linked_measurement_count": len(measurements),
            "linked_measurement_ids": ";".join(measurements),
            "linked_stable_ids": ";".join(stable_ids),
        })
    claims_rows.sort(key=lambda row: (
        row["i_atlas_sample"], row["variable"], float(row["value"]),
        row["source_kind"], row["source_id"],
    ))
    claim_fields = [
        "variable", "i_atlas_sample", "value", "source_kind", "scope",
        "association", "source_id", "source_url", "reference", "note",
        "linked_measurement_count", "linked_measurement_ids", "linked_stable_ids",
    ]
    with claims_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=claim_fields)
        writer.writeheader()
        writer.writerows(claims_rows)

    rows_by_parent: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows_out:
        rows_by_parent[str(row["i_atlas_sample"])].append(row)
    parent_rows = []
    for parent, linked_rows in sorted(rows_by_parent.items()):
        feo_records = [claim_links[key] for key in parent_claim_keys[parent]["FeO_wt_percent"]]
        is_records = [claim_links[key] for key in parent_claim_keys[parent]["Is_FeO"]]
        direct = [record for record in feo_records if record["association"] == "exact_relab_measurement"]
        parent_level = [record for record in feo_records if record["association"] == "apollo_parent"]

        def values(records):
            return sorted({float(record["value"]) for record in records})

        def mean_or_blank(numbers):
            return statistics.fmean(numbers) if numbers else ""

        feo_values = values(feo_records)
        direct_values = values(direct)
        parent_values = values(parent_level)
        maturity_values = values(is_records)
        parent_rows.append({
            "i_atlas_sample": parent,
            "relab_measurement_count": len(linked_rows),
            "stable_ids": ";".join(str(row["id"]) for row in linked_rows),
            "relab_measurement_ids": ";".join(str(row["relab_measurement_id"]) for row in linked_rows),
            "atlas_sample_types": ";".join(sorted({str(row["atlas_sample_type"]) for row in linked_rows if row["atlas_sample_type"]})),
            "atlas_lithologies": ";".join(sorted({str(row["atlas_lithology"]) for row in linked_rows if row["atlas_lithology"]})),
            "feo_claim_count": len(feo_records),
            "feo_unique_value_count": len(feo_values),
            "feo_mean_wt_percent": mean_or_blank(feo_values),
            "feo_min_wt_percent": min(feo_values) if feo_values else "",
            "feo_max_wt_percent": max(feo_values) if feo_values else "",
            "feo_values_wt_percent": ";".join(format(value, ".10g") for value in feo_values),
            "feo_direct_claim_count": len(direct),
            "feo_direct_mean_wt_percent": mean_or_blank(direct_values),
            "feo_parent_claim_count": len(parent_level),
            "feo_parent_mean_wt_percent": mean_or_blank(parent_values),
            "is_feo_claim_count": len(is_records),
            "is_feo_mean": mean_or_blank(maturity_values),
            "is_feo_min": min(maturity_values) if maturity_values else "",
            "is_feo_max": max(maturity_values) if maturity_values else "",
            "is_feo_values": ";".join(format(value, ".10g") for value in maturity_values),
        })
    parent_fields = list(parent_rows[0])
    with parent_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=parent_fields)
        writer.writeheader()
        writer.writerows(parent_rows)

    input_paths = {
        "comprehensive_table": args.comprehensive_table,
        "m3_spectra": args.m3_spectra,
        "atlas_pdf_feo": args.atlas_pdf_feo,
        "adjudications": args.adjudications,
        "missing_resolutions": args.missing_resolutions,
        "manual_feo": args.manual_feo,
        "source_chemistry_index": args.source_chemistry_index,
        "existing_is_feo": args.existing_is_feo,
    }
    inputs = {
        label: {"path": str(path.resolve()), "sha256": sha256(path)}
        for label, path in input_paths.items() if path is not None and path.exists()
    }
    parent_ids = {row["i_atlas_sample"] for row in rows_out}
    feo_rows = [row for row in rows_out if row["feo_wt_percent_claim_count"]]
    maturity_rows = [row for row in rows_out if row["is_feo_claim_count"]]
    manifest = {
        "schema": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs,
        "input_claim_counts": input_counts,
        "output_csv": str(args.output_csv.resolve()),
        "claims_csv": str(claims_path.resolve()),
        "parent_summary_csv": str(parent_path.resolve()),
        "unique_claim_rows": len(claims_rows),
        "row_count": len(rows_out),
        "unique_relab_measurements": len({row["relab_measurement_id"] for row in rows_out}),
        "unique_apollo_parent_count": len(parent_ids),
        "rows_with_feo": len(feo_rows),
        "parents_with_feo": len({row["i_atlas_sample"] for row in feo_rows}),
        "rows_with_is_feo": len(maturity_rows),
        "parents_with_is_feo": len({row["i_atlas_sample"] for row in maturity_rows}),
        "feo_target_scope_counts": dict(Counter(row["feo_target_scope"] for row in rows_out)),
        "required_m3_bands": list(BANDS),
        "m3_band_centroids_nm": {str(band): centroids.get(band) for band in BANDS},
        "feo_target": "arithmetic mean of unique accepted direct and parent-level claims",
        "is_feo_target": "arithmetic mean of unique reviewed source claims when available",
        "excluded_measurements_without_all_74_bands": excluded_no_74,
        "excluded_measurements_without_feo_or_is_feo": excluded_no_target,
        "excluded_inconsistent_identity": identity_warnings,
        "baseline_definitions": {
            "standard_lucey": "origin=(0.08,1.19); FeO=17.427*H-7.565",
            "origin_changed_lucey": (
                f"origin=({ORIGIN_CHANGED_X0},{ORIGIN_CHANGED_Y0}); "
                f"FeO={ORIGIN_CHANGED_A}*H+{ORIGIN_CHANGED_B}"
            ),
            "principal_H": "-atan((R950/R750-y0)/(R750-x0))",
            "atan2_2pi_H": "(-atan2(R950/R750-y0,R750-x0)) mod 2*pi",
        },
        "warnings": warnings,
        "important_notes": [
            "One row is one unique RELAB measurement; stable comprehensive-table IDs are retained.",
            "All spectra from one Apollo parent must stay in one validation fold.",
            "Parent-level chemistry is provisional for child/fraction spectra and is explicitly marked.",
            "Unreviewed PDF candidates and component/mineral-only FeO values are excluded.",
            "Is/FeO is ferromagnetic-resonance maturity, not a reflectance ratio.",
        ],
    }
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Ground-truth rows: {len(rows_out)}")
    print(f"Apollo parent groups: {manifest['unique_apollo_parent_count']}")
    print(f"Rows with FeO: {manifest['rows_with_feo']}")
    print(f"Parents with FeO: {manifest['parents_with_feo']}")
    print(f"Rows with Is/FeO: {manifest['rows_with_is_feo']}")
    print(f"Parents with Is/FeO: {manifest['parents_with_is_feo']}")
    print(f"Target scopes: {manifest['feo_target_scope_counts']}")
    print(f"Excluded without all 74 bands: {excluded_no_74}")
    print(f"Excluded without accepted FeO or Is/FeO: {excluded_no_target}")
    print(f"Dataset: {args.output_csv}")
    print(f"Claims: {claims_path}")
    print(f"Parent summary: {parent_path}")
    print(f"Manifest: {args.manifest_json}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
