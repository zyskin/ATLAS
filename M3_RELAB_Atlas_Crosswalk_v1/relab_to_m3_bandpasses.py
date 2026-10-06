#!/usr/bin/env python3
"""Forward-model native RELAB spectra through the 85 M3 global bandpasses.

The native RELAB point table is never modified.  The output is one long CSV
with one row per RELAB measurement and M3 band.  A band is reported only when
the native spectrum covers the requested fraction of that band's response.

For band b the reported value is

    sum_lambda B_b(lambda) R(lambda) / sum_lambda B_b(lambda)

over wavelengths for which the RELAB spectrum can be interpolated.  The
coverage fraction records the fraction of the full official bandpass used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from itertools import groupby
from pathlib import Path

import numpy as np


VERSION = "relab-to-m3-bandpasses-1.0"
OUTPUT_COLUMNS = [
    "measurement_id",
    "m3_band",
    "m3_response_centroid_nm",
    "m3_reflectance",
    "bandpass_coverage_fraction",
    "usable",
    "native_point_count",
    "native_segment_count",
    "native_wavelength_min_nm",
    "native_wavelength_max_nm",
    "interpolation",
    "minimum_coverage_required",
    "maximum_interpolation_gap_nm",
]


def clean(value: object) -> str:
    return " ".join(str(value or "").split())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def band_number(name: str) -> int:
    import re

    match = re.search(r"(\d+)\s*$", name)
    if not match:
        raise ValueError(f"Cannot obtain M3 band number from CSV column {name!r}")
    return int(match.group(1))


def load_bandpasses(path: Path) -> tuple[np.ndarray, list[int], np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Empty bandpass CSV: {path}") from exc
        if len(header) < 2:
            raise ValueError("Bandpass CSV must contain wavelength plus band columns")
        wavelength_name = clean(header[0]).casefold()
        if "wavelength" not in wavelength_name:
            raise ValueError(f"First bandpass column must be wavelength, found {header[0]!r}")
        numbered = [(band_number(name), index, name) for index, name in enumerate(header[1:], 1)]
        numbered.sort()
        bands = [item[0] for item in numbered]
        if bands != list(range(1, 86)):
            raise ValueError(f"Expected M3 bands 1..85; found {bands}")

        wavelengths: list[float] = []
        rows: list[list[float]] = []
        for line_number, row in enumerate(reader, 2):
            if not row or not any(cell.strip() for cell in row):
                continue
            if len(row) != len(header):
                raise ValueError(
                    f"Bandpass CSV line {line_number}: {len(row)} fields; expected {len(header)}"
                )
            wavelengths.append(float(row[0]))
            rows.append([float(row[item[1]]) for item in numbered])

    grid = np.asarray(wavelengths, dtype=np.float64)
    weights = np.asarray(rows, dtype=np.float64)
    if grid.shape != (2701,) or weights.shape != (2701, 85):
        raise ValueError(
            f"Expected bandpass shapes (2701,) and (2701,85); got {grid.shape} and {weights.shape}"
        )
    if not np.all(np.diff(grid) > 0):
        raise ValueError("Bandpass wavelengths are not strictly increasing")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Bandpass weights must be finite and nonnegative")
    totals = weights.sum(axis=0)
    if np.any(totals <= 0):
        raise ValueError("At least one M3 bandpass has zero total weight")
    centroids = (weights * grid[:, None]).sum(axis=0) / totals
    return grid, bands, weights, centroids


def split_native_segments(rows: list[dict[str, str]]) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    parsed: list[tuple[int, float, float]] = []
    for row in rows:
        try:
            index = int(row["point_index"])
            wavelength = float(row["wavelength_nm"])
            reflectance = float(row["reflectance"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(wavelength) and math.isfinite(reflectance):
            parsed.append((index, wavelength, reflectance))
    parsed.sort(key=lambda value: value[0])

    raw_segments: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    previous: float | None = None
    for _, wavelength, reflectance in parsed:
        if previous is not None and wavelength <= previous and current:
            raw_segments.append(current)
            current = []
        current.append((wavelength, reflectance))
        previous = wavelength
    if current:
        raw_segments.append(current)

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for segment in raw_segments:
        if len(segment) < 2:
            continue
        x = np.asarray([item[0] for item in segment], dtype=np.float64)
        y = np.asarray([item[1] for item in segment], dtype=np.float64)
        if np.all(np.diff(x) > 0):
            segments.append((x, y))
    return segments, len(parsed)


def interpolation_mask(x: np.ndarray, grid: np.ndarray, maximum_gap_nm: float) -> np.ndarray:
    inside = (grid >= x[0]) & (grid <= x[-1])
    if not np.any(inside):
        return inside
    positions = np.searchsorted(x, grid, side="left")
    exact = (positions < len(x)) & inside
    exact[exact] &= x[positions[exact]] == grid[exact]
    bracketed = inside & ~exact & (positions > 0) & (positions < len(x))
    allowed = exact.copy()
    allowed[bracketed] = (
        x[positions[bracketed]] - x[positions[bracketed] - 1] <= maximum_gap_nm
    )
    return allowed


def interpolate_segments(
    segments: list[tuple[np.ndarray, np.ndarray]],
    grid: np.ndarray,
    method: str,
    maximum_gap_nm: float,
) -> np.ndarray:
    accumulated = np.zeros(grid.shape, dtype=np.float64)
    counts = np.zeros(grid.shape, dtype=np.int32)
    pchip_class = None
    if method == "pchip":
        try:
            from scipy.interpolate import PchipInterpolator
        except ImportError as exc:
            raise RuntimeError(
                "SciPy is required for --interpolation pchip. Install scipy or use "
                "--interpolation linear."
            ) from exc
        pchip_class = PchipInterpolator

    for x, y in segments:
        allowed = interpolation_mask(x, grid, maximum_gap_nm)
        if not np.any(allowed):
            continue
        if method == "linear":
            values = np.interp(grid[allowed], x, y)
        else:
            values = pchip_class(x, y, extrapolate=False)(grid[allowed])
        finite = np.isfinite(values)
        indices = np.flatnonzero(allowed)[finite]
        accumulated[indices] += values[finite]
        counts[indices] += 1

    result = np.full(grid.shape, np.nan, dtype=np.float64)
    available = counts > 0
    result[available] = accumulated[available] / counts[available]
    return result


def forward_model(
    reconstructed: np.ndarray,
    weights: np.ndarray,
    minimum_coverage: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    available = np.isfinite(reconstructed)
    totals = weights.sum(axis=0)
    covered_weights = weights[available, :].sum(axis=0)
    coverage = covered_weights / totals
    values = np.full(weights.shape[1], np.nan, dtype=np.float64)
    usable = (coverage >= minimum_coverage) & (covered_weights > 0)
    if np.any(usable):
        numerators = (weights[available, :] * reconstructed[available, None]).sum(axis=0)
        values[usable] = numerators[usable] / covered_weights[usable]
    return values, coverage, usable


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectral-points", type=Path, required=True)
    parser.add_argument("--bandpass-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--manifest-json",
        type=Path,
        help="Default: OUTPUT_CSV with suffix .manifest.json",
    )
    parser.add_argument(
        "--interpolation", choices=("pchip", "linear"), default="pchip",
        help="Shape-preserving PCHIP is the default; linear requires only NumPy.",
    )
    parser.add_argument("--minimum-coverage", type=float, default=0.99)
    parser.add_argument("--maximum-interpolation-gap-nm", type=float, default=50.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0 < args.minimum_coverage <= 1:
        print("ERROR: --minimum-coverage must be in (0,1]", file=sys.stderr)
        return 2
    if args.maximum_interpolation_gap_nm <= 0:
        print("ERROR: --maximum-interpolation-gap-nm must be positive", file=sys.stderr)
        return 2
    for path in (args.spectral_points, args.bandpass_csv):
        if not path.is_file():
            print(f"ERROR: input file not found: {path}", file=sys.stderr)
            return 2
    manifest_path = args.manifest_json or args.output_csv.with_suffix(".manifest.json")
    if not args.overwrite:
        existing = [path for path in (args.output_csv, manifest_path) if path.exists()]
        if existing:
            print(f"ERROR: output exists; use --overwrite: {existing[0]}", file=sys.stderr)
            return 2

    try:
        grid, bands, weights, centroids = load_bandpasses(args.bandpass_csv)
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output_csv.with_name(args.output_csv.name + ".part")
        measurements = 0
        output_rows = 0
        usable_rows = 0
        seen: set[str] = set()
        warnings: list[str] = []

        with args.spectral_points.open(newline="", encoding="utf-8-sig") as source, temporary.open(
            "w", newline="", encoding="utf-8"
        ) as destination:
            reader = csv.DictReader(source)
            required = {"measurement_id", "point_index", "wavelength_nm", "reflectance"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Missing spectral columns: {', '.join(sorted(missing))}")
            writer = csv.DictWriter(destination, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()

            for measurement_id, grouped in groupby(reader, key=lambda row: clean(row.get("measurement_id"))):
                if not measurement_id:
                    continue
                if measurement_id in seen:
                    raise ValueError(
                        f"Measurement {measurement_id!r} occurs in non-contiguous blocks; "
                        "sort the spectral table by measurement_id and point_index first."
                    )
                seen.add(measurement_id)
                rows = list(grouped)
                segments, point_count = split_native_segments(rows)
                measurements += 1
                if not segments:
                    warnings.append(f"{measurement_id}: fewer than two usable native points")
                    reconstructed = np.full(grid.shape, np.nan)
                    native_min = native_max = ""
                else:
                    reconstructed = interpolate_segments(
                        segments, grid, args.interpolation, args.maximum_interpolation_gap_nm
                    )
                    native_min = min(float(x[0]) for x, _ in segments)
                    native_max = max(float(x[-1]) for x, _ in segments)
                values, coverage, usable = forward_model(
                    reconstructed, weights, args.minimum_coverage
                )
                for index, band in enumerate(bands):
                    is_usable = bool(usable[index])
                    writer.writerow(
                        {
                            "measurement_id": measurement_id,
                            "m3_band": band,
                            "m3_response_centroid_nm": format(float(centroids[index]), ".10g"),
                            "m3_reflectance": format(float(values[index]), ".10g") if is_usable else "",
                            "bandpass_coverage_fraction": format(float(coverage[index]), ".10g"),
                            "usable": "yes" if is_usable else "no",
                            "native_point_count": point_count,
                            "native_segment_count": len(segments),
                            "native_wavelength_min_nm": native_min,
                            "native_wavelength_max_nm": native_max,
                            "interpolation": args.interpolation,
                            "minimum_coverage_required": args.minimum_coverage,
                            "maximum_interpolation_gap_nm": args.maximum_interpolation_gap_nm,
                        }
                    )
                    output_rows += 1
                    usable_rows += int(is_usable)
                if measurements % 100 == 0:
                    print(f"Processed {measurements} RELAB measurements", file=sys.stderr)

        temporary.replace(args.output_csv)
        manifest = {
            "schema": "relab-m3-forward-model-manifest-1.0",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "program": VERSION,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "formula": "sum(B_band(lambda)*R_relab(lambda))/sum(B_band(lambda)) over covered wavelengths",
            "interpolation": args.interpolation,
            "minimum_coverage": args.minimum_coverage,
            "maximum_interpolation_gap_nm": args.maximum_interpolation_gap_nm,
            "overlapping_native_segments": "interpolated values are averaged at wavelengths covered by multiple segments",
            "extrapolation": "none",
            "bandpass_grid_nm": [float(grid[0]), float(grid[-1])],
            "band_count": len(bands),
            "measurement_count": measurements,
            "output_row_count": output_rows,
            "usable_output_row_count": usable_rows,
            "inputs": {
                "spectral_points": str(args.spectral_points.resolve()),
                "spectral_points_sha256": sha256_file(args.spectral_points),
                "bandpass_csv": str(args.bandpass_csv.resolve()),
                "bandpass_csv_sha256": sha256_file(args.bandpass_csv),
            },
            "output_csv": str(args.output_csv.resolve()),
            "output_csv_sha256": sha256_file(args.output_csv),
            "warnings": warnings[:100],
            "warning_count": len(warnings),
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"RELAB measurements: {measurements}")
        print(f"M3 rows written: {output_rows}")
        print(f"Usable M3 values: {usable_rows}")
        print(f"Output: {args.output_csv}")
        print(f"Manifest: {manifest_path}")
        return 0
    except (OSError, ValueError, csv.Error, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
