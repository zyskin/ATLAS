#!/usr/bin/env python3
"""Compare matched RELAB and Adams spectra at M3 band-centre wavelengths.

The program uses natural cubic splines (second derivative zero at both ends)
for both spectra, evaluates them at the response-weighted centres of the 85 M3
global bands, and computes RELAB/Adams reflectance ratios.

Unless --pair-map is supplied, spectra are paired through a shared Apollo
parent sample.  This is a parent-level association, not proof that the exact
physical subsample, grain-size fraction, or preparation is identical.

Outputs
-------
relab_adams_ratios_by_sample_band.csv
    One row per matched RELAB/Adams pair and usable M3 band.  Pair-level means
    across bands are repeated on each row for convenient filtering.

relab_adams_ratio_summary.csv
    Mean and sample standard deviation for every M3 band, plus the distribution
    of the per-pair means across bands.  Results are reported both pair-weighted
    and with each Apollo parent receiving equal weight.

relab_adams_ratio_summary_short_child.csv
relab_adams_ratio_summary_short_parent.csv
    Compact band, mean-ratio, standard-deviation tables.  A child is one
    matched RELAB/Adams spectral pair.  Parent results first average children
    within each Apollo parent, then give every parent equal weight.

relab_adams_ratio_total_child.csv
relab_adams_ratio_total_parent.csv
    One-row summaries of the per-match mean coefficients over bands.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from relab_m3_suitability import (
    REQUIRED_MAX_NM,
    REQUIRED_MIN_NM,
    assess_relab_suitability,
)

try:
    from scipy.interpolate import CubicSpline
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "SciPy is required. In the active environment run: "
        "python -m pip install scipy"
    ) from exc


DETAIL_NAME = "relab_adams_ratios_by_sample_band.csv"
VERSION = "4.0"
SUMMARY_NAME = "relab_adams_ratio_summary.csv"
SHORT_CHILD_NAME = "relab_adams_ratio_summary_short_child.csv"
SHORT_PARENT_NAME = "relab_adams_ratio_summary_short_parent.csv"
TOTAL_CHILD_NAME = "relab_adams_ratio_total_child.csv"
TOTAL_PARENT_NAME = "relab_adams_ratio_total_parent.csv"
PLOT_CHILD_NAME = "relab_adams_ratio_by_band_child.png"
PLOT_PARENT_NAME = "relab_adams_ratio_by_band_parent.png"


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


def format_number(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return format(value, ".17g")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def require_columns(path: Path, rows: list[dict[str, str]], required: set[str]) -> None:
    fields = set(rows[0]) if rows else set()
    missing = sorted(required - fields)
    if missing:
        raise ValueError(f"{path} lacks required columns: {', '.join(missing)}")


def m3_band_centres(path: Path) -> list[tuple[int, float]]:
    """Return response-weighted M3 band centres from the official BPF CSV."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header or len(header) < 2:
            raise ValueError(f"Invalid M3 bandpass CSV: {path}")

        numbered: list[tuple[int, int]] = []
        for column_index, name in enumerate(header[1:], 1):
            match = re.search(r"(\d+)\s*$", clean(name))
            if not match:
                raise ValueError(f"Cannot read band number from column {name!r}")
            numbered.append((int(match.group(1)), column_index))
        numbered.sort()
        if [band for band, _ in numbered] != list(range(1, 86)):
            raise ValueError("The bandpass CSV must contain M3 bands 1 through 85")

        wavelengths: list[float] = []
        responses: list[list[float]] = []
        for line_number, row in enumerate(reader, 2):
            if not row or not any(clean(cell) for cell in row):
                continue
            if len(row) != len(header):
                raise ValueError(
                    f"Bandpass CSV row {line_number} has {len(row)} fields; "
                    f"expected {len(header)}"
                )
            wavelengths.append(float(row[0]))
            responses.append([float(row[index]) for _, index in numbered])

    wavelength = np.asarray(wavelengths, dtype=np.float64)
    response = np.asarray(responses, dtype=np.float64)
    if wavelength.ndim != 1 or response.shape != (len(wavelength), 85):
        raise ValueError("Unexpected M3 bandpass array dimensions")
    if not np.all(np.diff(wavelength) > 0):
        raise ValueError("M3 bandpass wavelengths are not strictly increasing")
    if np.any(~np.isfinite(response)) or np.any(response < 0):
        raise ValueError("M3 band responses must be finite and nonnegative")
    totals = response.sum(axis=0)
    if np.any(totals <= 0):
        raise ValueError("One or more M3 bands have zero integrated response")
    centres = (response * wavelength[:, None]).sum(axis=0) / totals
    return [(band, float(centres[band - 1])) for band in range(1, 86)]


def load_relab_index(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv_rows(path)
    require_columns(path, rows, {"measurement_id"})
    return {canonical_id(row["measurement_id"]): row for row in rows}


def is_standard_geometry(row: dict[str, str], tolerance: float) -> bool:
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


def load_adams_rows(path: Path, body: str) -> list[dict[str, str]]:
    rows = read_csv_rows(path)
    require_columns(
        path,
        rows,
        {
            "body",
            "apollo_sample_id",
            "adams_specimen_id",
            "corrected_measurement_id",
            "corrected_file",
        },
    )
    if body.casefold() == "all":
        return rows
    return [row for row in rows if clean(row.get("body")).casefold() == body.casefold()]


def load_relab_crosswalk(path: Path) -> list[dict[str, str]]:
    rows = read_csv_rows(path)
    require_columns(
        path,
        rows,
        {"measurement_id", "relab_specimen_id", "relab_specimen_name", "apollo_parent_sample"},
    )
    return rows


def build_pairs(
    adams_rows: list[dict[str, str]],
    relab_rows: list[dict[str, str]],
    relab_index: dict[str, dict[str, str]],
    geometry: str,
    geometry_tolerance: float,
    required_coverage_min_nm: float,
    required_coverage_max_nm: float,
    pair_map_path: Path | None,
) -> list[dict[str, object]]:
    adams_by_measurement = {
        canonical_id(row["corrected_measurement_id"]): row for row in adams_rows
    }
    relab_by_measurement = {canonical_id(row["measurement_id"]): row for row in relab_rows}

    pairs: list[dict[str, object]] = []
    if pair_map_path:
        map_rows = read_csv_rows(pair_map_path)
        require_columns(
            pair_map_path,
            map_rows,
            {"adams_measurement_id", "relab_measurement_id"},
        )
        for map_row in map_rows:
            adams_id = canonical_id(map_row["adams_measurement_id"])
            relab_id = canonical_id(map_row["relab_measurement_id"])
            if adams_id not in adams_by_measurement:
                raise ValueError(f"Pair map refers to unknown Adams measurement {adams_id}")
            if relab_id not in relab_by_measurement:
                raise ValueError(f"Pair map refers to unknown RELAB measurement {relab_id}")
            adams = adams_by_measurement[adams_id]
            relab = relab_by_measurement[relab_id]
            parent = canonical_id(
                map_row.get("apollo_parent_id")
                or relab.get("apollo_parent_sample")
                or adams.get("apollo_sample_id")
            )
            pairs.append(
                {
                    "parent": parent,
                    "adams": adams,
                    "relab": relab,
                    "match_basis": clean(map_row.get("match_basis")) or "explicit_pair_map",
                }
            )
    else:
        adams_by_parent: dict[str, list[dict[str, str]]] = defaultdict(list)
        relab_by_parent: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in adams_rows:
            parent = canonical_id(row.get("apollo_sample_id"))
            if parent:
                adams_by_parent[parent].append(row)
        for row in relab_rows:
            parent = canonical_id(row.get("apollo_parent_sample"))
            measurement = canonical_id(row.get("measurement_id"))
            if not parent or not measurement:
                continue
            if geometry == "standard":
                index_row = relab_index.get(measurement, {})
                if not is_standard_geometry(index_row, geometry_tolerance):
                    continue
            relab_by_parent[parent].append(row)
        for parent in sorted(set(adams_by_parent) & set(relab_by_parent)):
            for adams in adams_by_parent[parent]:
                for relab in relab_by_parent[parent]:
                    pairs.append(
                        {
                            "parent": parent,
                            "adams": adams,
                            "relab": relab,
                            "match_basis": "shared_apollo_parent",
                        }
                    )

    filtered: list[dict[str, object]] = []
    for pair in pairs:
        relab = pair["relab"]
        measurement = canonical_id(relab["measurement_id"])
        if measurement not in relab_index:
            raise ValueError(f"RELAB index lacks measurement {measurement}")
        suitability = assess_relab_suitability(
            relab_index[measurement],
            required_coverage_min_nm,
            required_coverage_max_nm,
        )
        if not suitability["suitable"]:
            continue
        if geometry == "standard" and not is_standard_geometry(
            relab_index[measurement], geometry_tolerance
        ):
            continue
        filtered.append(pair)

    filtered.sort(
        key=lambda pair: (
            canonical_id(pair["parent"]),
            canonical_id(pair["adams"]["corrected_measurement_id"]),
            canonical_id(pair["relab"]["measurement_id"]),
        )
    )
    return filtered


def load_relab_points(
    path: Path, wanted_measurements: set[str]
) -> dict[str, list[tuple[int, float, float]]]:
    result: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = {"measurement_id", "point_index", "wavelength_nm", "reflectance"} - set(
            reader.fieldnames or []
        )
        if missing:
            raise ValueError(f"{path} lacks required columns: {', '.join(sorted(missing))}")
        for row in reader:
            measurement = canonical_id(row.get("measurement_id"))
            if measurement not in wanted_measurements:
                continue
            wavelength = number(row.get("wavelength_nm"))
            reflectance = number(row.get("reflectance"))
            try:
                point_index = int(clean(row.get("point_index")))
            except ValueError:
                continue
            if wavelength is not None and reflectance is not None:
                result[measurement].append((point_index, wavelength, reflectance))
    missing_measurements = sorted(wanted_measurements - set(result))
    if missing_measurements:
        preview = ", ".join(missing_measurements[:10])
        raise ValueError(f"No spectral points found for RELAB measurements: {preview}")
    return result


def split_segments(rows: list[tuple[int, float, float]]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Preserve native order and split where wavelength ceases to increase."""
    ordered = sorted(rows, key=lambda item: item[0])
    raw_segments: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    previous: float | None = None
    for _, wavelength, reflectance in ordered:
        if previous is not None and wavelength <= previous and current:
            raw_segments.append(current)
            current = []
        current.append((wavelength, reflectance))
        previous = wavelength
    if current:
        raw_segments.append(current)

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for raw in raw_segments:
        grouped: dict[float, list[float]] = defaultdict(list)
        for wavelength, reflectance in raw:
            grouped[wavelength].append(reflectance)
        x = np.asarray(sorted(grouped), dtype=np.float64)
        y = np.asarray([np.mean(grouped[value]) for value in x], dtype=np.float64)
        if len(x) >= 2 and np.all(np.diff(x) > 0):
            segments.append((x, y))
    return segments


def read_adams_spectrum(path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    start = 0
    for index, line in enumerate(lines):
        lower = line.casefold()
        if "wavelength" in lower and "reflectance" in lower:
            start = index + 1
            break
    points: list[tuple[int, float, float]] = []
    for line_number, line in enumerate(lines[start:], start + 1):
        fields = re.split(r"[\s,;]+", line.strip())
        if len(fields) < 2:
            continue
        try:
            wavelength = float(fields[0])
            reflectance = float(fields[1])
        except ValueError:
            continue
        if math.isfinite(wavelength) and math.isfinite(reflectance):
            points.append((line_number, wavelength, reflectance))
    if len(points) < 2:
        raise ValueError(f"Could not read at least two spectral points from {path}")
    maximum = max(point[1] for point in points)
    if maximum < 50.0:
        points = [(index, wavelength * 1000.0, value) for index, wavelength, value in points]
    return split_segments(points)


def natural_spline_values(
    segments: list[tuple[np.ndarray, np.ndarray]], targets_nm: np.ndarray
) -> np.ndarray:
    sums = np.zeros(len(targets_nm), dtype=np.float64)
    counts = np.zeros(len(targets_nm), dtype=np.int32)
    for x, y in segments:
        inside = (targets_nm >= x[0]) & (targets_nm <= x[-1])
        if not np.any(inside):
            continue
        spline = CubicSpline(x, y, bc_type="natural", extrapolate=False)
        values = np.asarray(spline(targets_nm[inside]), dtype=np.float64)
        finite = np.isfinite(values)
        indices = np.flatnonzero(inside)[finite]
        sums[indices] += values[finite]
        counts[indices] += 1
    result = np.full(len(targets_nm), np.nan, dtype=np.float64)
    available = counts > 0
    result[available] = sums[available] / counts[available]
    return result


def sample_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    script_root = Path(__file__).resolve().parent
    parser.add_argument("--project-root", type=Path, default=script_root)
    parser.add_argument("--relab-points", type=Path)
    parser.add_argument("--relab-index", type=Path)
    parser.add_argument("--relab-crosswalk", type=Path)
    parser.add_argument("--adams-crosswalk", type=Path)
    parser.add_argument("--adams-corrected-dir", type=Path)
    parser.add_argument("--bandpass-csv", type=Path)
    parser.add_argument(
        "--pair-map",
        type=Path,
        help=(
            "Optional exact-pair CSV with adams_measurement_id and "
            "relab_measurement_id. Without it, all spectra sharing an Apollo "
            "parent are compared."
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--body", default="Apollo", help="Apollo (default) or all")
    parser.add_argument(
        "--relab-geometry",
        choices=("standard", "any"),
        default="standard",
        help="standard keeps only 30/0/30 degree RELAB spectra (default)",
    )
    parser.add_argument("--geometry-tolerance-deg", type=float, default=0.01)
    parser.add_argument(
        "--required-coverage-min-nm",
        type=float,
        default=REQUIRED_MIN_NM,
        help="RELAB spectrum must begin at or before this wavelength (default: 461)",
    )
    parser.add_argument(
        "--required-coverage-max-nm",
        type=float,
        default=REQUIRED_MAX_NM,
        help="RELAB spectrum must end at or after this wavelength (default: 2500)",
    )
    parser.add_argument(
        "--ratio-direction",
        choices=("relab_over_adams", "adams_over_relab"),
        default="relab_over_adams",
    )
    parser.add_argument("--minimum-denominator", type=float, default=1e-12)
    parser.add_argument(
        "--append-detail",
        action="store_true",
        help=(
            "Append the current per-sample/per-band rows to the detailed CSV. "
            "By default it is overwritten. All derived summaries are always overwritten."
        ),
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Write separate child-weighted and parent-weighted PNG plots.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Accepted for compatibility; overwriting is now the default.",
    )
    return parser.parse_args(argv)


def resolved_paths(args: argparse.Namespace) -> dict[str, Path]:
    root = args.project_root.resolve()
    adams_root = root / "data" / "adams-20220223T225406Z-001" / "adams"
    return {
        "relab_points": args.relab_points
        or root / "relab_raw_native_2026-10-04" / "relab_spectral_points.csv",
        "relab_index": args.relab_index
        or root / "relab_raw_native_2026-10-04" / "relab_spectrum_index.csv",
        "relab_crosswalk": args.relab_crosswalk
        or root / "crosswalk_output_v2_2026-10-03" / "relab_atlas_crosswalk.csv",
        "adams_crosswalk": args.adams_crosswalk
        or root / "data" / "adams_id_lists" / "adams_atlas_crosswalk_validated.csv",
        "adams_corrected_dir": args.adams_corrected_dir
        or adams_root / "wavelength_corrected_data",
        "bandpass_csv": args.bandpass_csv
        or root / "Calibration" / "M3G20081211_RDN_BPF.csv",
        "output_dir": args.output_dir or root / "data" / "relab_adams_ratio_v1",
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = resolved_paths(args)
    input_names = (
        "relab_points",
        "relab_index",
        "relab_crosswalk",
        "adams_crosswalk",
        "bandpass_csv",
    )
    for name in input_names:
        if not paths[name].is_file():
            print(f"ERROR: {name} not found: {paths[name]}", file=sys.stderr)
            return 2
    if not paths["adams_corrected_dir"].is_dir():
        print(
            f"ERROR: Adams corrected-data directory not found: {paths['adams_corrected_dir']}",
            file=sys.stderr,
        )
        return 2
    if args.pair_map and not args.pair_map.is_file():
        print(f"ERROR: pair map not found: {args.pair_map}", file=sys.stderr)
        return 2
    if args.minimum_denominator <= 0 or args.geometry_tolerance_deg < 0:
        print("ERROR: tolerances must be nonnegative and denominator positive", file=sys.stderr)
        return 2
    if args.required_coverage_min_nm >= args.required_coverage_max_nm:
        print(
            "ERROR: --required-coverage-min-nm must be smaller than "
            "--required-coverage-max-nm",
            file=sys.stderr,
        )
        return 2

    output_dir = paths["output_dir"]
    detail_path = output_dir / DETAIL_NAME
    summary_path = output_dir / SUMMARY_NAME
    short_child_path = output_dir / SHORT_CHILD_NAME
    short_parent_path = output_dir / SHORT_PARENT_NAME
    total_child_path = output_dir / TOTAL_CHILD_NAME
    total_parent_path = output_dir / TOTAL_PARENT_NAME
    plot_child_path = output_dir / PLOT_CHILD_NAME
    plot_parent_path = output_dir / PLOT_PARENT_NAME

    try:
        centres = m3_band_centres(paths["bandpass_csv"])
        bands = np.asarray([band for band, _ in centres], dtype=np.int32)
        wavelengths = np.asarray([value for _, value in centres], dtype=np.float64)
        relab_index = load_relab_index(paths["relab_index"])
        relab_rows = load_relab_crosswalk(paths["relab_crosswalk"])
        adams_rows = load_adams_rows(paths["adams_crosswalk"], args.body)
        pairs = build_pairs(
            adams_rows,
            relab_rows,
            relab_index,
            args.relab_geometry,
            args.geometry_tolerance_deg,
            args.required_coverage_min_nm,
            args.required_coverage_max_nm,
            args.pair_map,
        )
        if not pairs:
            raise ValueError("No RELAB/Adams pairs satisfy the selected matching and geometry rules")
        wanted_relab = {
            canonical_id(pair["relab"]["measurement_id"]) for pair in pairs
        }
        relab_points = load_relab_points(paths["relab_points"], wanted_relab)

        relab_values: dict[str, np.ndarray] = {}
        for measurement in sorted(wanted_relab):
            relab_values[measurement] = natural_spline_values(
                split_segments(relab_points[measurement]), wavelengths
            )

        adams_values: dict[str, np.ndarray] = {}
        for pair in pairs:
            adams = pair["adams"]
            measurement = canonical_id(adams["corrected_measurement_id"])
            if measurement in adams_values:
                continue
            file_path = paths["adams_corrected_dir"] / clean(adams["corrected_file"])
            if not file_path.is_file():
                raise ValueError(f"Adams corrected spectrum not found: {file_path}")
            adams_values[measurement] = natural_spline_values(
                read_adams_spectrum(file_path), wavelengths
            )
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    detailed: list[dict[str, object]] = []
    pair_summaries: list[dict[str, object]] = []
    for pair_number, pair in enumerate(pairs, 1):
        parent = canonical_id(pair["parent"])
        adams = pair["adams"]
        relab = pair["relab"]
        adams_id = canonical_id(adams["corrected_measurement_id"])
        relab_id = canonical_id(relab["measurement_id"])
        index_row = relab_index[relab_id]
        a_values = adams_values[adams_id]
        r_values = relab_values[relab_id]

        valid = np.isfinite(a_values) & np.isfinite(r_values)
        if args.ratio_direction == "relab_over_adams":
            valid &= np.abs(a_values) >= args.minimum_denominator
        else:
            valid &= np.abs(r_values) >= args.minimum_denominator
        valid_indices = np.flatnonzero(valid)
        if len(valid_indices) == 0:
            continue

        ratio_ra = r_values[valid_indices] / a_values[valid_indices]
        ratio_ar = a_values[valid_indices] / r_values[valid_indices]
        selected = ratio_ra if args.ratio_direction == "relab_over_adams" else ratio_ar
        finite = np.isfinite(selected) & np.isfinite(ratio_ra) & np.isfinite(ratio_ar)
        valid_indices = valid_indices[finite]
        ratio_ra = ratio_ra[finite]
        ratio_ar = ratio_ar[finite]
        selected = selected[finite]
        if len(valid_indices) == 0:
            continue

        pair_mean = float(np.mean(selected))
        pair_std = float(np.std(selected, ddof=1)) if len(selected) >= 2 else None
        pair_id = f"P{pair_number:04d}"
        pair_summaries.append(
            {
                "pair_id": pair_id,
                "parent": parent,
                "mean": pair_mean,
            }
        )
        for local_index, band_index in enumerate(valid_indices):
            detailed.append(
                {
                    "pair_id": pair_id,
                    "apollo_parent_id": parent,
                    "match_basis": pair["match_basis"],
                    "adams_measurement_id": adams_id,
                    "adams_specimen_id": clean(adams.get("adams_specimen_id")),
                    "adams_corrected_file": clean(adams.get("corrected_file")),
                    "relab_measurement_id": relab_id,
                    "relab_specimen_id": clean(relab.get("relab_specimen_id")),
                    "relab_specimen_name": clean(relab.get("relab_specimen_name")),
                    "relab_instrument": clean(index_row.get("instrument_names")),
                    "relab_incidence_angle_deg": clean(index_row.get("incidence_angle_deg")),
                    "relab_emission_angle_deg": clean(index_row.get("emission_angle_deg")),
                    "relab_phase_angle_deg": clean(index_row.get("phase_angle_deg")),
                    "relab_data_wavelength_min_nm": clean(
                        index_row.get("data_wavelength_min_nm")
                    ),
                    "relab_data_wavelength_max_nm": clean(
                        index_row.get("data_wavelength_max_nm")
                    ),
                    "required_coverage_min_nm": args.required_coverage_min_nm,
                    "required_coverage_max_nm": args.required_coverage_max_nm,
                    "suitable_for_adams_m3_comparison": "yes",
                    "m3_band": int(bands[band_index]),
                    "m3_response_centroid_nm": float(wavelengths[band_index]),
                    "relab_reflectance": float(r_values[band_index]),
                    "adams_reflectance": float(a_values[band_index]),
                    "ratio_relab_over_adams": float(ratio_ra[local_index]),
                    "ratio_adams_over_relab": float(ratio_ar[local_index]),
                    "selected_ratio_direction": args.ratio_direction,
                    "selected_ratio": float(selected[local_index]),
                    "pair_usable_band_count": len(valid_indices),
                    "pair_mean_ratio_over_bands": pair_mean,
                    "pair_std_ratio_over_bands": pair_std,
                    "interpolation": "natural_cubic_spline",
                }
            )

    if not detailed:
        print("ERROR: no common finite M3 band-centre values were produced", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    detail_fields = list(detailed[0])
    detail_mode = "a" if args.append_detail and detail_path.exists() else "w"
    write_detail_header = detail_mode == "w" or detail_path.stat().st_size == 0
    if detail_mode == "a" and detail_path.stat().st_size > 0:
        with detail_path.open(newline="", encoding="utf-8-sig") as existing:
            existing_header = next(csv.reader(existing), [])
        if existing_header != detail_fields:
            print(
                "ERROR: existing detailed CSV has a different schema; "
                "rerun without --append-detail to overwrite it",
                file=sys.stderr,
            )
            return 2
    with detail_path.open(detail_mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=detail_fields)
        if write_detail_header:
            writer.writeheader()
        for row in detailed:
            writer.writerow(
                {
                    key: format_number(value) if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )

    summary_rows: list[dict[str, object]] = []
    for band in sorted({int(row["m3_band"]) for row in detailed}):
        band_rows = [row for row in detailed if int(row["m3_band"]) == band]
        wavelength = float(band_rows[0]["m3_response_centroid_nm"])
        pair_ratios = [float(row["selected_ratio"]) for row in band_rows]

        by_parent: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in band_rows:
            by_parent[str(row["apollo_parent_id"])].append(row)
        parent_ratios = [
            float(np.mean([float(row["selected_ratio"]) for row in rows]))
            for rows in by_parent.values()
        ]
        for weighting, ratios in (
            ("matched_child_pair", pair_ratios),
            ("apollo_parent_equal", parent_ratios),
        ):
            summary_rows.append(
                {
                    "summary_scope": "m3_band",
                    "m3_band": band,
                    "m3_response_centroid_nm": wavelength,
                    "weighting": weighting,
                    "ratio_direction": args.ratio_direction,
                    "n_values": len(ratios),
                    "n_matched_pairs": len(band_rows),
                    "n_apollo_parents": len(by_parent),
                    "mean_ratio": float(np.mean(ratios)),
                    "sample_std_ratio": sample_std(ratios),
                }
            )

    pair_means = [float(row["mean"]) for row in pair_summaries]
    by_parent_summary: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in pair_summaries:
        by_parent_summary[str(row["parent"])].append(row)
    parent_means = [
        float(np.mean([float(row["mean"]) for row in rows]))
        for rows in by_parent_summary.values()
    ]
    for weighting, values in (
        ("matched_child_pair", pair_means),
        ("apollo_parent_equal", parent_means),
    ):
        summary_rows.append(
            {
                "summary_scope": "sample_mean_over_bands",
                "m3_band": "",
                "m3_response_centroid_nm": "",
                "weighting": weighting,
                "ratio_direction": args.ratio_direction,
                "n_values": len(values),
                "n_matched_pairs": len(pair_summaries),
                "n_apollo_parents": len(by_parent_summary),
                "mean_ratio": float(np.mean(values)),
                "sample_std_ratio": sample_std(values),
            }
        )

    summary_fields = list(summary_rows[0])
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(
                {
                    key: format_number(value) if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )

    def short_rows(weighting: str) -> list[dict[str, object]]:
        return [
            {
                "m3_band": row["m3_band"],
                "mean_ratio": row["mean_ratio"],
                "sample_std_ratio": row["sample_std_ratio"],
            }
            for row in summary_rows
            if row["summary_scope"] == "m3_band" and row["weighting"] == weighting
        ]

    def total_row(weighting: str) -> dict[str, object]:
        matches = [
            row
            for row in summary_rows
            if row["summary_scope"] == "sample_mean_over_bands"
            and row["weighting"] == weighting
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one total row for {weighting}")
        return {
            "mean_ratio": matches[0]["mean_ratio"],
            "sample_std_ratio": matches[0]["sample_std_ratio"],
        }

    def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
        fields = list(rows[0])
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: format_number(value) if isinstance(value, float) else value
                        for key, value in row.items()
                    }
                )

    child_short = short_rows("matched_child_pair")
    parent_short = short_rows("apollo_parent_equal")
    child_total = total_row("matched_child_pair")
    parent_total = total_row("apollo_parent_equal")
    write_rows(short_child_path, child_short)
    write_rows(short_parent_path, parent_short)
    write_rows(total_child_path, [child_total])
    write_rows(total_parent_path, [parent_total])

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print(
                "ERROR: --plot requires matplotlib; run python -m pip install matplotlib",
                file=sys.stderr,
            )
            return 2

        def make_plot(
            rows: list[dict[str, object]],
            total: dict[str, object],
            title: str,
            output: Path,
        ) -> None:
            x = np.asarray([int(row["m3_band"]) for row in rows], dtype=np.int32)
            mean = np.asarray([float(row["mean_ratio"]) for row in rows], dtype=np.float64)
            std = np.asarray(
                [
                    float(row["sample_std_ratio"])
                    if row["sample_std_ratio"] is not None
                    else np.nan
                    for row in rows
                ],
                dtype=np.float64,
            )
            overall = float(total["mean_ratio"])
            ratio_label = (
                "RELAB reflectance / Adams reflectance"
                if args.ratio_direction == "relab_over_adams"
                else "Adams reflectance / RELAB reflectance"
            )
            figure, axis = plt.subplots(figsize=(11, 6))
            axis.plot(x, mean, color="tab:blue", linewidth=2.0, label="Mean ratio")
            finite = np.isfinite(std)
            axis.fill_between(
                x,
                mean - std,
                mean + std,
                where=finite,
                color="tab:blue",
                alpha=0.22,
                label="Mean ± one sample SD",
            )
            axis.axhline(
                overall,
                color="tab:red",
                linestyle="--",
                linewidth=1.8,
                label=f"Overall mean of per-match band means = {overall:.5g}",
            )
            axis.set_xlabel("M3 band number")
            axis.set_ylabel(ratio_label)
            axis.set_title(title)
            axis.grid(True, alpha=0.25)
            axis.legend()
            figure.tight_layout()
            figure.savefig(output, dpi=180)
            plt.close(figure)

        direction_title = (
            "RELAB/Adams" if args.ratio_direction == "relab_over_adams" else "Adams/RELAB"
        )
        make_plot(
            child_short,
            child_total,
            f"{direction_title} coefficient by M3 band: matched-child weighting",
            plot_child_path,
        )
        make_plot(
            parent_short,
            parent_total,
            f"{direction_title} coefficient by M3 band: equal Apollo-parent weighting",
            plot_parent_path,
        )

    print(f"Matched pairs used: {len(pair_summaries)}")
    print(f"Apollo parents used: {len(by_parent_summary)}")
    print(f"Detailed sample-band rows: {len(detailed)}")
    print(f"Interpolation: natural cubic spline (endpoint second derivatives = 0)")
    print(
        "Required RELAB wavelength coverage: "
        f"{args.required_coverage_min_nm:g}-{args.required_coverage_max_nm:g} nm"
    )
    print(f"Ratio direction summarized: {args.ratio_direction}")
    if not args.pair_map:
        print("CAUTION: matches are by shared Apollo parent, not exact physical subsample")
    print(f"Detailed CSV: {detail_path}")
    print(f"Summary CSV: {summary_path}")
    print(f"Short child CSV: {short_child_path}")
    print(f"Short parent CSV: {short_parent_path}")
    print(f"Total child CSV: {total_child_path}")
    print(f"Total parent CSV: {total_parent_path}")
    if args.plot:
        print(f"Child plot: {plot_child_path}")
        print(f"Parent plot: {plot_parent_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

