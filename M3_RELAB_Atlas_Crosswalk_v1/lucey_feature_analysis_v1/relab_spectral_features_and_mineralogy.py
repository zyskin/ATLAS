#!/usr/bin/env python3
"""Compute S, D1, D2 and mean reflectivity for every RELAB spectrum.

The input spectra are the official-bandpass-convolved long RELAB/M3 table.
Natural cubic splines are used only to evaluate smooth diagnostic continua and
absorption searches between the M3 response centroids.  They are not used to
create the M3-equivalent band values themselves.

The output mineral labels are deliberately named ``*_like``.  They are
editable exploratory rules, not mineral abundance estimates.  Metadata keyword
labels are recorded independently so agreement and disagreement can be audited.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.interpolate import CubicSpline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m3-spectra", required=True, type=Path)
    parser.add_argument("--rules-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--spectrum-index", type=Path)
    parser.add_argument("--comprehensive-table", type=Path)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--atlas-enriched", type=Path)
    parser.add_argument("--maximum-band", type=int, default=74)
    parser.add_argument("--dense-step-nm", type=float, default=2.0)
    return parser.parse_args()


def clean(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return " ".join(str(value).split())


def truthy(value: object) -> bool:
    return clean(value).casefold() in {"yes", "true", "1", "y"}


def first_nonempty(series: pd.Series) -> str:
    for value in series:
        text = clean(value)
        if text:
            return text
    return ""


def unique_join(series: pd.Series, separator: str = " | ") -> str:
    values: list[str] = []
    seen: set[str] = set()
    for value in series:
        text = clean(value)
        if text and text not in seen:
            seen.add(text)
            values.append(text)
    return separator.join(values)


def load_optional_csv(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def build_metadata(
    spectrum_index: pd.DataFrame,
    comprehensive: pd.DataFrame,
    ground_truth: pd.DataFrame,
    atlas: pd.DataFrame,
) -> pd.DataFrame:
    records: dict[str, dict[str, Any]] = {}

    if not spectrum_index.empty:
        if "measurement_id" not in spectrum_index:
            raise ValueError("Spectrum index lacks measurement_id")
        wanted = [
            "specimen_id", "specimen_name", "specimen_description",
            "specimen_collection_location", "specimen_type", "material_origin",
            "material_type", "material_subtypes", "incidence_angle_deg",
            "emission_angle_deg", "phase_angle_deg",
        ]
        for row in spectrum_index.to_dict("records"):
            measurement = clean(row.get("measurement_id"))
            if measurement:
                records.setdefault(measurement, {}).update(
                    {column: clean(row.get(column)) for column in wanted}
                )

    if not comprehensive.empty:
        if "relab_measurement_id" not in comprehensive:
            raise ValueError("Comprehensive table lacks relab_measurement_id")
        grouped = comprehensive.groupby("relab_measurement_id", dropna=False)
        for measurement, group in grouped:
            measurement = clean(measurement)
            if not measurement:
                continue
            record = records.setdefault(measurement, {})
            for column in (
                "i_atlas_sample", "j_child", "atlas_sample_type", "atlas_lithology",
                "age_or_treatment_notes", "comments", "crosswalk_match_type",
                "crosswalk_needs_review", "atlas_detail_url", "atlas_pdf_url",
            ):
                if column in group:
                    record[column] = unique_join(group[column])

    if not ground_truth.empty:
        if "relab_measurement_id" not in ground_truth:
            raise ValueError("Ground-truth table lacks relab_measurement_id")
        for row in ground_truth.to_dict("records"):
            measurement = clean(row.get("relab_measurement_id"))
            if not measurement:
                continue
            record = records.setdefault(measurement, {})
            for column in (
                "id", "i_atlas_sample", "j_child", "feo_target_scope",
                "feo_wt_percent_mean", "feo_wt_percent_min", "feo_wt_percent_max",
                "feo_direct_wt_percent_mean", "is_feo_mean", "atlas_sample_type",
                "atlas_lithology", "crosswalk_needs_review",
            ):
                if column in row and clean(row.get(column)):
                    record[column] = row.get(column)

    atlas_by_id: dict[str, dict[str, Any]] = {}
    if not atlas.empty and "atlas_sample_id" in atlas:
        atlas_by_id = {
            clean(row.get("atlas_sample_id")): row for row in atlas.to_dict("records")
        }
    for record in records.values():
        atlas_id = clean(record.get("i_atlas_sample"))
        atlas_row = atlas_by_id.get(atlas_id, {})
        for source, destination in (
            ("sample_type", "atlas_enriched_sample_type"),
            ("lithology", "atlas_enriched_lithology"),
            ("description", "atlas_enriched_description"),
            ("atlas_pdf_url", "atlas_enriched_pdf_url"),
        ):
            if clean(atlas_row.get(source)):
                record[destination] = clean(atlas_row.get(source))

    rows = [{"measurement_id": key, **value} for key, value in records.items()]
    return pd.DataFrame(rows)


def continuum(
    spline: CubicSpline,
    wavelength: np.ndarray,
    left: float,
    right: float,
) -> np.ndarray:
    r_left = float(spline(left))
    r_right = float(spline(right))
    return r_left + (r_right - r_left) * (wavelength - left) / (right - left)


def band_depth(
    spline: CubicSpline,
    continuum_left: float,
    search_left: float,
    search_right: float,
    continuum_right: float,
    step: float,
) -> tuple[float, float]:
    grid = np.arange(search_left, search_right + 0.5 * step, step)
    reflectance = np.asarray(spline(grid), dtype=float)
    cont = continuum(spline, grid, continuum_left, continuum_right)
    valid = np.isfinite(reflectance) & np.isfinite(cont) & (cont > 0)
    if not np.any(valid):
        return math.nan, math.nan
    depth = 1.0 - reflectance[valid] / cont[valid]
    index = int(np.nanargmax(depth))
    return float(depth[index]), float(grid[valid][index])


def features_for_spectrum(
    wavelength: np.ndarray,
    reflectance: np.ndarray,
    config: dict[str, Any],
    step: float,
) -> dict[str, Any]:
    order = np.argsort(wavelength)
    wavelength = np.asarray(wavelength[order], dtype=float)
    reflectance = np.asarray(reflectance[order], dtype=float)
    unique, unique_index = np.unique(wavelength, return_index=True)
    wavelength = unique
    reflectance = reflectance[unique_index]
    finite = np.isfinite(wavelength) & np.isfinite(reflectance)
    wavelength = wavelength[finite]
    reflectance = reflectance[finite]
    if wavelength.size < 8 or np.any(np.diff(wavelength) <= 0):
        raise ValueError("insufficient strictly increasing finite M3 points")

    w = config["feature_wavelengths_nm"]
    spline = CubicSpline(wavelength, reflectance, bc_type="natural", extrapolate=False)

    def covered(left: float, right: float) -> bool:
        return wavelength[0] <= left and wavelength[-1] >= right

    result: dict[str, Any] = {
        "available_wavelength_min_nm": float(wavelength[0]),
        "available_wavelength_max_nm": float(wavelength[-1]),
        "S_per_micron": math.nan,
        "D1": math.nan,
        "D1_minimum_nm": math.nan,
        "D2": math.nan,
        "D2_minimum_nm": math.nan,
        "mean_reflectivity_750_2500": math.nan,
        "mean_reflectivity_available_interval": math.nan,
        "mean_available_left_nm": math.nan,
        "mean_available_right_nm": math.nan,
        "R750_spline": math.nan,
        "R1500_spline": math.nan,
        "S_available": False,
        "D1_available": False,
        "D2_available": False,
        "mean_750_2500_available": False,
    }

    if covered(w["slope_left"], w["slope_right"]):
        r_left = float(spline(w["slope_left"]))
        r_right = float(spline(w["slope_right"]))
        delta_micrometres = (w["slope_right"] - w["slope_left"]) / 1000.0
        result["R750_spline"] = r_left
        result["R1500_spline"] = r_right
        result["S_per_micron"] = (
            (r_right - r_left) / (r_left * delta_micrometres)
            if r_left > 0 else math.nan
        )
        result["S_available"] = math.isfinite(result["S_per_micron"])

    if covered(w["band1_continuum_left"], w["band1_continuum_right"]):
        result["D1"], result["D1_minimum_nm"] = band_depth(
            spline,
            w["band1_continuum_left"], w["band1_search_left"],
            w["band1_search_right"], w["band1_continuum_right"], step,
        )
        result["D1_available"] = math.isfinite(result["D1"])

    if covered(w["band2_continuum_left"], w["band2_continuum_right"]):
        result["D2"], result["D2_minimum_nm"] = band_depth(
            spline,
            w["band2_continuum_left"], w["band2_search_left"],
            w["band2_search_right"], w["band2_continuum_right"], step,
        )
        result["D2_available"] = math.isfinite(result["D2"])

    mean_left = max(float(w["mean_left"]), float(wavelength[0]))
    mean_right = min(float(w["mean_right"]), float(wavelength[-1]))
    if mean_right > mean_left:
        mean_grid = np.arange(mean_left, mean_right + 0.5 * step, step)
        mean_grid[-1] = min(mean_grid[-1], mean_right)
        mean_values = np.asarray(spline(mean_grid), dtype=float)
        finite = np.isfinite(mean_values)
        if finite.sum() >= 2:
            grid = mean_grid[finite]
            values = mean_values[finite]
            result["mean_reflectivity_available_interval"] = float(
                trapezoid(values, grid) / (grid[-1] - grid[0])
            )
            result["mean_available_left_nm"] = float(grid[0])
            result["mean_available_right_nm"] = float(grid[-1])

    if covered(w["mean_left"], w["mean_right"]):
        result["mean_reflectivity_750_2500"] = result[
            "mean_reflectivity_available_interval"
        ]
        result["mean_750_2500_available"] = math.isfinite(
            result["mean_reflectivity_750_2500"]
        )

    full = all(
        bool(result[key])
        for key in ("S_available", "D1_available", "D2_available", "mean_750_2500_available")
    )
    result["feature_coverage"] = "full" if full else "partial"
    missing = [
        label for label, key in (
            ("S", "S_available"),
            ("D1", "D1_available"),
            ("D2", "D2_available"),
            ("mean_750_2500", "mean_750_2500_available"),
        ) if not result[key]
    ]
    result["missing_features"] = ";".join(missing)
    return result


def preliminary_class(features: dict[str, float], config: dict[str, Any]) -> tuple[str, str]:
    t = config["thresholds"]
    d1 = features["D1"]
    d2 = features["D2"]
    mean = features["mean_reflectivity_750_2500"]
    slope = features["S_per_micron"]
    if not all(math.isfinite(value) for value in (d1, d2, mean, slope)):
        return (
            "unclassified_insufficient_coverage",
            "one or more required full-range features are unavailable; no mineral label assigned",
        )
    if d1 >= t["deep_D1"] and d2 >= t["deep_D2"]:
        return "pyroxene_like", "both 1-micrometre and 2-micrometre absorptions are strong"
    if d1 <= t["weak_D1"] and d2 >= t["deep_D2"]:
        return "spinel_like", "weak 1-micrometre and strong 2-micrometre absorption"
    if d1 >= t["deep_D1"] and d2 <= t["weak_D2"]:
        return "olivine_like", "strong broad 1-micrometre and weak 2-micrometre absorption"
    if d1 <= t["weak_D1"] and d2 <= t["weak_D2"] and mean >= t["bright_mean_reflectivity"]:
        return "plagioclase_like", "weak absorptions and high mean reflectivity"
    if (
        d1 <= t["soil_max_D1"] and d2 <= t["soil_max_D2"]
        and mean <= t["dark_mean_reflectivity"]
        and slope >= t["soil_min_S_per_micron"]
    ):
        return "mature_soil_like", "weak absorptions, low reflectivity, nonnegative red slope"
    return "mixed_or_uncertain", "no preliminary endmember rule is satisfied"


KEYWORDS: dict[str, tuple[str, ...]] = {
    "pyroxene": ("pyroxene", "orthopyroxene", "clinopyroxene", "pigeonite", "augite", " opx", " cpx"),
    "olivine": ("olivine", "dunite", "forsterite", "fayalite"),
    "plagioclase": ("plagioclase", "feldspar", "anorthosite", "anorthositic"),
    "spinel": ("spinel", "chromite"),
    "soil_regolith": ("soil", "regolith", "unconsolidated mixture"),
    "mature": ("mature", "submature"),
    "basalt": ("basalt", "mare"),
    "breccia": ("breccia",),
}


def metadata_labels(record: dict[str, Any]) -> tuple[str, str]:
    fields = [
        "specimen_name", "specimen_description", "material_type", "material_subtypes",
        "atlas_sample_type", "atlas_lithology", "comments",
        "atlas_enriched_sample_type", "atlas_enriched_lithology",
        "atlas_enriched_description", "age_or_treatment_notes",
    ]
    text = " ".join(clean(record.get(field)) for field in fields).casefold()
    labels = [label for label, words in KEYWORDS.items() if any(word in text for word in words)]
    evidence = " | ".join(
        f"{field}={clean(record.get(field))}" for field in fields if clean(record.get(field))
    )
    return ";".join(labels), evidence


def compare_labels(spectral: str, metadata: str) -> str:
    labels = set(filter(None, metadata.split(";")))
    if not labels:
        return "metadata_unavailable"
    expected = {
        "pyroxene_like": "pyroxene",
        "olivine_like": "olivine",
        "plagioclase_like": "plagioclase",
        "spinel_like": "spinel",
        "mature_soil_like": "soil_regolith",
    }.get(spectral)
    if expected is None:
        return "spectral_rule_uncertain"
    return "agrees_with_metadata_keyword" if expected in labels else "differs_or_mixture_needs_review"


def main() -> int:
    args = parse_args()
    config = json.loads(args.rules_json.read_text(encoding="utf-8"))
    spectra = pd.read_csv(args.m3_spectra, low_memory=False)
    required = {
        "measurement_id", "m3_band", "m3_response_centroid_nm",
        "m3_reflectance", "usable",
    }
    missing = required.difference(spectra.columns)
    if missing:
        raise ValueError(f"M3 spectra table lacks: {', '.join(sorted(missing))}")
    spectra["m3_band"] = pd.to_numeric(spectra["m3_band"], errors="coerce")
    spectra["m3_response_centroid_nm"] = pd.to_numeric(
        spectra["m3_response_centroid_nm"], errors="coerce"
    )
    spectra["m3_reflectance"] = pd.to_numeric(spectra["m3_reflectance"], errors="coerce")
    usable = spectra["usable"].map(truthy)
    spectra = spectra.loc[
        usable
        & (spectra["m3_band"] <= args.maximum_band)
        & np.isfinite(spectra["m3_response_centroid_nm"])
        & np.isfinite(spectra["m3_reflectance"])
    ].copy()

    metadata = build_metadata(
        load_optional_csv(args.spectrum_index),
        load_optional_csv(args.comprehensive_table),
        load_optional_csv(args.ground_truth),
        load_optional_csv(args.atlas_enriched),
    )
    metadata_by_measurement = {
        clean(row.get("measurement_id")): row for row in metadata.to_dict("records")
    }

    output_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for measurement, group in spectra.groupby("measurement_id", sort=True):
        measurement = clean(measurement)
        base: dict[str, Any] = {"measurement_id": measurement}
        record = metadata_by_measurement.get(measurement, {})
        base.update(record)
        try:
            feature = features_for_spectrum(
                group["m3_response_centroid_nm"].to_numpy(float),
                group["m3_reflectance"].to_numpy(float),
                config,
                args.dense_step_nm,
            )
            spectral_label, reason = preliminary_class(feature, config)
            base.update(feature)
            base["preliminary_spectral_class"] = spectral_label
            base["spectral_class_reason"] = reason
            base["feature_status"] = (
                "ok_full" if feature["feature_coverage"] == "full" else "ok_partial"
            )
        except ValueError as exc:
            failures.append(f"{measurement}: {exc}")
            base.update({
                "S_per_micron": math.nan, "D1": math.nan, "D1_minimum_nm": math.nan,
                "D2": math.nan, "D2_minimum_nm": math.nan,
                "mean_reflectivity_750_2500": math.nan,
                "R750_spline": math.nan, "R1500_spline": math.nan,
                "mean_reflectivity_available_interval": math.nan,
                "mean_available_left_nm": math.nan,
                "mean_available_right_nm": math.nan,
                "available_wavelength_min_nm": math.nan,
                "available_wavelength_max_nm": math.nan,
                "S_available": False, "D1_available": False,
                "D2_available": False, "mean_750_2500_available": False,
                "feature_coverage": "none", "missing_features": "S;D1;D2;mean_750_2500",
                "preliminary_spectral_class": "unclassified_insufficient_coverage",
                "spectral_class_reason": str(exc), "feature_status": "insufficient_coverage",
            })
        labels, evidence = metadata_labels(base)
        base["metadata_keyword_labels"] = labels
        base["metadata_evidence_text"] = evidence
        base["spectral_metadata_comparison"] = compare_labels(
            base["preliminary_spectral_class"], labels
        )
        base["usable_m3_band_count"] = int(len(group))
        output_rows.append(base)

    output = pd.DataFrame(output_rows)
    preferred = [
        "measurement_id", "id", "i_atlas_sample", "j_child", "specimen_id",
        "specimen_name", "specimen_description", "material_type", "material_subtypes",
        "atlas_sample_type", "atlas_lithology", "atlas_enriched_description",
        "feo_target_scope", "feo_wt_percent_mean", "feo_wt_percent_min",
        "feo_wt_percent_max", "feo_direct_wt_percent_mean", "is_feo_mean",
        "S_per_micron", "D1", "D1_minimum_nm", "D2", "D2_minimum_nm",
        "mean_reflectivity_750_2500", "mean_reflectivity_available_interval",
        "mean_available_left_nm", "mean_available_right_nm",
        "available_wavelength_min_nm", "available_wavelength_max_nm",
        "R750_spline", "R1500_spline", "S_available", "D1_available",
        "D2_available", "mean_750_2500_available", "feature_coverage",
        "missing_features",
        "preliminary_spectral_class", "spectral_class_reason",
        "metadata_keyword_labels", "spectral_metadata_comparison",
        "metadata_evidence_text", "usable_m3_band_count", "feature_status",
        "crosswalk_needs_review", "atlas_detail_url", "atlas_pdf_url",
    ]
    ordered = [column for column in preferred if column in output]
    ordered += [column for column in output.columns if column not in ordered]
    output = output[ordered]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)

    summary_path = args.summary_json or args.output_csv.with_suffix(".summary.json")
    summary = {
        "schema": "relab-spectral-features-mineralogy-1.0",
        "input_m3_spectra": str(args.m3_spectra),
        "rules": str(args.rules_json),
        "maximum_band": args.maximum_band,
        "measurement_count": int(len(output)),
        "full_feature_count": int((output["feature_status"] == "ok_full").sum()),
        "partial_feature_count": int((output["feature_status"] == "ok_partial").sum()),
        "feature_failure_count": int((output["feature_status"] == "insufficient_coverage").sum()),
        "spectral_class_counts": output["preliminary_spectral_class"].value_counts().to_dict(),
        "comparison_counts": output["spectral_metadata_comparison"].value_counts().to_dict(),
        "warning": config.get("warning"),
        "failures_first_100": failures[:100],
        "output_csv": str(args.output_csv),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Measurements written: {len(output)}")
    print(f"Full feature records: {summary['full_feature_count']}")
    print(f"Partial feature records: {summary['partial_feature_count']}")
    print(f"Output: {args.output_csv}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
