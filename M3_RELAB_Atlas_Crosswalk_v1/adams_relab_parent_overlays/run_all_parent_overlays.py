#!/usr/bin/env python3
"""Plot every validated Adams spectrum against all RELAB spectra at its Apollo parent.

One PNG is written for each Adams row whose Apollo parent occurs in the RELAB--Atlas
crosswalk.  Every RELAB measurement attached to that parent is plotted in blue, while
one coefficient-rescaled Adams spectrum is plotted in red.  The large RELAB spectral
points CSV is scanned only once.  Two CSV manifests record the plot-level selection and
every RELAB measurement included in every plot.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


VERSION = "1.0"


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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def require_columns(path: Path, rows: list[dict[str, str]], required: set[str]) -> None:
    fields = set(rows[0]) if rows else set()
    missing = sorted(required - fields)
    if missing:
        raise ValueError(f"{path} lacks required columns: {', '.join(missing)}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=script_dir.parent,
        help="M3_RELAB_Atlas_Crosswalk_v1 directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "plots",
        help="Directory for PNG files and generated manifests",
    )
    parser.add_argument(
        "--coefficient",
        choices=("parent", "child", "none"),
        default="parent",
        help=(
            "Adams multiplier: parent is parent-balanced (default), child is the "
            "legacy name for pair-weighted, and none leaves Adams unscaled"
        ),
    )
    parser.add_argument("--coefficient-summary", type=Path)
    parser.add_argument("--bandpass-csv", type=Path)
    parser.add_argument("--wavelength-min", type=float, default=350.0)
    parser.add_argument("--wavelength-max", type=float, default=2600.0)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--require-atlas-page",
        action="store_true",
        help=(
            "Require atlas_join_status=exact_atlas_id. By default a known Apollo "
            "parent such as 14003 is retained even if absent from the local Atlas index."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not redraw PNG files that already exist; manifests are still rewritten",
    )
    return parser.parse_args(argv)


def resolved_paths(args: argparse.Namespace) -> dict[str, Path]:
    root = args.project_root.resolve()
    coefficient_dir = root / "data" / "relab_adams_ratio_v1"
    coefficient_name = (
        "relab_adams_ratio_summary_short_parent.csv"
        if args.coefficient == "parent"
        else "relab_adams_ratio_summary_short_child.csv"
    )
    return {
        "root": root,
        "adams_crosswalk": root
        / "data"
        / "adams_id_lists"
        / "adams_atlas_crosswalk_validated.csv",
        "adams_corrected_dir": root
        / "data"
        / "adams-20220223T225406Z-001"
        / "adams"
        / "wavelength_corrected_data",
        "relab_crosswalk": root
        / "crosswalk_output_v2_2026-10-03"
        / "relab_atlas_crosswalk.csv",
        "relab_index": root
        / "relab_raw_native_2026-10-04"
        / "relab_spectrum_index.csv",
        "relab_points": root
        / "relab_raw_native_2026-10-04"
        / "relab_spectral_points.csv",
        "coefficient_summary": (
            args.coefficient_summary.resolve()
            if args.coefficient_summary
            else coefficient_dir / coefficient_name
        ),
        "bandpass_csv": (
            args.bandpass_csv.resolve()
            if args.bandpass_csv
            else root / "Calibration" / "M3G20081211_RDN_BPF.csv"
        ),
        "output_dir": args.output_dir.resolve(),
    }


def m3_band_centres(path: Path) -> dict[int, float]:
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
                    f"Bandpass row {line_number} has {len(row)} fields; "
                    f"expected {len(header)}"
                )
            wavelengths.append(float(row[0]))
            responses.append([float(row[index]) for _, index in numbered])

    wavelength = np.asarray(wavelengths, dtype=np.float64)
    response = np.asarray(responses, dtype=np.float64)
    if response.shape != (len(wavelength), 85):
        raise ValueError("Unexpected M3 bandpass dimensions")
    if not np.all(np.diff(wavelength) > 0):
        raise ValueError("M3 bandpass wavelengths are not strictly increasing")
    totals = response.sum(axis=0)
    if np.any(~np.isfinite(response)) or np.any(response < 0) or np.any(totals <= 0):
        raise ValueError("M3 band responses must be finite, nonnegative, and nonzero")
    centres = (response * wavelength[:, None]).sum(axis=0) / totals
    return {band: float(centres[band - 1]) for band in range(1, 86)}


def load_coefficient(summary_path: Path, bandpass_path: Path) -> tuple[np.ndarray, np.ndarray]:
    centres = m3_band_centres(bandpass_path)
    rows = read_csv_rows(summary_path)
    require_columns(summary_path, rows, {"m3_band", "mean_ratio"})
    by_band: dict[int, float] = {}
    for line_number, row in enumerate(rows, 2):
        try:
            band = int(clean(row.get("m3_band")))
            value = float(clean(row.get("mean_ratio")))
        except ValueError as exc:
            raise ValueError(f"Invalid coefficient at {summary_path}:{line_number}") from exc
        if band not in centres or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid coefficient for M3 band {band}: {value!r}")
        if band in by_band:
            raise ValueError(f"Duplicate coefficient for M3 band {band}")
        by_band[band] = value
    if len(by_band) < 2:
        raise ValueError(f"At least two coefficients are required in {summary_path}")
    bands = sorted(by_band)
    return (
        np.asarray([centres[band] for band in bands], dtype=np.float64),
        np.asarray([by_band[band] for band in bands], dtype=np.float64),
    )


def read_adams_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    wavelengths: list[float] = []
    reflectances: list[float] = []
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    start = 0
    for index, line in enumerate(lines):
        if "wavelength" in line.casefold() and "reflectance" in line.casefold():
            start = index + 1
            break
    for line in lines[start:]:
        fields = re.split(r"[\s,;]+", line.strip())
        if len(fields) < 2:
            continue
        try:
            wavelength = float(fields[0])
            reflectance = float(fields[1])
        except ValueError:
            continue
        if math.isfinite(wavelength) and math.isfinite(reflectance):
            wavelengths.append(wavelength)
            reflectances.append(reflectance)
    if len(wavelengths) < 2:
        raise ValueError(f"Could not read an Adams spectrum from {path}")
    x = np.asarray(wavelengths, dtype=np.float64)
    y = np.asarray(reflectances, dtype=np.float64)
    if float(np.max(x)) < 50.0:
        x = x * 1000.0
    order = np.argsort(x, kind="stable")
    return x[order], y[order]


def load_relab_points(
    path: Path, wanted: set[str]
) -> dict[str, list[tuple[int, float, float]]]:
    points: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"measurement_id", "point_index", "wavelength_nm", "reflectance"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} lacks required columns: {', '.join(sorted(missing))}")
        for row in reader:
            measurement = canonical_id(row.get("measurement_id"))
            if measurement not in wanted:
                continue
            wavelength = number(row.get("wavelength_nm"))
            reflectance = number(row.get("reflectance"))
            try:
                point_index = int(clean(row.get("point_index")))
            except ValueError:
                continue
            if wavelength is not None and reflectance is not None:
                points[measurement].append((point_index, wavelength, reflectance))
    return points


def split_segments(rows: list[tuple[int, float, float]]) -> list[tuple[np.ndarray, np.ndarray]]:
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
        if len(x) >= 2:
            segments.append((x, y))
    return segments


def safe_name(value: object) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", clean(value))
    return result.strip("._-") or "unknown"


def relationship_scope(rows: list[dict[str, str]]) -> str:
    types = {clean(row.get("match_type")) for row in rows}
    if types == {"sample_number"}:
        return "same_parent_exact_parent_record"
    if types == {"subsample_or_fraction"}:
        return "same_parent_child_or_fraction"
    if types == {"atlas_missing"}:
        return "same_parent_atlas_missing"
    return "same_parent_mixed_" + "_".join(sorted(types))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = resolved_paths(args)
    if args.wavelength_min >= args.wavelength_max:
        raise ValueError("--wavelength-min must be smaller than --wavelength-max")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")

    required_files = (
        "adams_crosswalk",
        "relab_crosswalk",
        "relab_index",
        "relab_points",
    )
    if args.coefficient != "none":
        required_files += ("coefficient_summary", "bandpass_csv")
    for name in required_files:
        if not paths[name].is_file():
            raise FileNotFoundError(f"{name} not found: {paths[name]}")
    if not paths["adams_corrected_dir"].is_dir():
        raise FileNotFoundError(
            f"Adams corrected-data directory not found: {paths['adams_corrected_dir']}"
        )

    adams_rows = read_csv_rows(paths["adams_crosswalk"])
    require_columns(
        paths["adams_crosswalk"],
        adams_rows,
        {
            "body",
            "apollo_sample_id",
            "adams_specimen_id",
            "corrected_measurement_id",
            "corrected_file",
            "atlas_join_status",
        },
    )
    relab_rows = read_csv_rows(paths["relab_crosswalk"])
    require_columns(
        paths["relab_crosswalk"],
        relab_rows,
        {
            "measurement_id",
            "relab_specimen_id",
            "relab_specimen_name",
            "apollo_parent_sample",
            "match_type",
        },
    )
    index_rows = read_csv_rows(paths["relab_index"])
    require_columns(paths["relab_index"], index_rows, {"measurement_id"})
    index_by_measurement = {
        canonical_id(row.get("measurement_id")): row for row in index_rows
    }

    relab_by_parent: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in relab_rows:
        parent = canonical_id(row.get("apollo_parent_sample"))
        measurement = canonical_id(row.get("measurement_id"))
        if parent and measurement:
            relab_by_parent[parent].append(row)
    for rows in relab_by_parent.values():
        rows.sort(key=lambda row: canonical_id(row.get("measurement_id")))

    selected_adams: list[dict[str, str]] = []
    for row in adams_rows:
        if clean(row.get("body")).casefold() != "apollo":
            continue
        parent = canonical_id(row.get("apollo_sample_id"))
        if not parent or parent not in relab_by_parent:
            continue
        if args.require_atlas_page and clean(row.get("atlas_join_status")) != "exact_atlas_id":
            continue
        selected_adams.append(row)
    selected_adams.sort(
        key=lambda row: (
            canonical_id(row.get("apollo_sample_id")),
            canonical_id(row.get("corrected_measurement_id")),
        )
    )
    if not selected_adams:
        raise ValueError("No Adams rows have a RELAB-linked Apollo parent")

    wanted_relab = {
        canonical_id(row.get("measurement_id"))
        for adams in selected_adams
        for row in relab_by_parent[canonical_id(adams.get("apollo_sample_id"))]
    }
    print(
        f"Reading one RELAB points table for {len(wanted_relab)} measurement IDs: "
        f"{paths['relab_points']}"
    )
    relab_points = load_relab_points(paths["relab_points"], wanted_relab)

    coefficient_wavelength: np.ndarray | None = None
    coefficient_value: np.ndarray | None = None
    if args.coefficient != "none":
        coefficient_wavelength, coefficient_value = load_coefficient(
            paths["coefficient_summary"], paths["bandpass_csv"]
        )

    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_manifest: list[dict[str, object]] = []
    relab_manifest: list[dict[str, object]] = []
    plots_written = 0
    plots_skipped = 0

    for plot_number, adams in enumerate(selected_adams, 1):
        parent = canonical_id(adams.get("apollo_sample_id"))
        adams_id = canonical_id(adams.get("adams_specimen_id"))
        corrected_id = canonical_id(adams.get("corrected_measurement_id"))
        corrected_file = clean(adams.get("corrected_file"))
        adams_path = paths["adams_corrected_dir"] / corrected_file
        if not adams_path.is_file():
            raise FileNotFoundError(f"Adams corrected spectrum not found: {adams_path}")
        adams_x, adams_y = read_adams_spectrum(adams_path)
        if coefficient_wavelength is not None and coefficient_value is not None:
            multiplier = np.interp(
                adams_x,
                coefficient_wavelength,
                coefficient_value,
                left=coefficient_value[0],
                right=coefficient_value[-1],
            )
            plotted_adams = adams_y * multiplier
        else:
            plotted_adams = adams_y

        parent_relab = relab_by_parent[parent]
        scope = relationship_scope(parent_relab)
        filename = (
            f"Apollo_{safe_name(parent)}_{safe_name(adams_id)}_{safe_name(corrected_id)}_"
            f"all_RELAB_{safe_name(args.coefficient)}_coefficient.png"
        )
        output_path = output_dir / filename

        segments_by_measurement = {
            canonical_id(row.get("measurement_id")): split_segments(
                relab_points.get(canonical_id(row.get("measurement_id")), [])
            )
            for row in parent_relab
        }
        available_measurements = {
            measurement
            for measurement, segments in segments_by_measurement.items()
            if segments
        }
        plotted_measurements = len(available_measurements)
        missing_measurements = [
            canonical_id(row.get("measurement_id"))
            for row in parent_relab
            if canonical_id(row.get("measurement_id")) not in available_measurements
        ]
        if args.skip_existing and output_path.exists():
            plots_skipped += 1
        else:
            figure, axis = plt.subplots(figsize=(12, 7))
            alpha = max(0.08, min(0.65, 7.0 / max(1, len(parent_relab))))
            for relab in parent_relab:
                measurement = canonical_id(relab.get("measurement_id"))
                segments = segments_by_measurement[measurement]
                if not segments:
                    continue
                for x, y in segments:
                    axis.plot(
                        x,
                        y,
                        color="#1f77b4",
                        linewidth=0.9,
                        alpha=alpha,
                        zorder=2,
                    )
            axis.plot(
                adams_x,
                plotted_adams,
                color="#d62728",
                linewidth=2.8,
                zorder=5,
            )
            specimen_count = len(
                {canonical_id(row.get("relab_specimen_id")) for row in parent_relab}
            )
            scaling_label = (
                "unscaled"
                if args.coefficient == "none"
                else (
                    "parent-balanced coefficient"
                    if args.coefficient == "parent"
                    else "pair-weighted coefficient (legacy name: child)"
                )
            )
            axis.set_title(
                f"Apollo parent {parent}: all RELAB spectra and Adams {adams_id} / {corrected_id}\n"
                f"{len(parent_relab)} RELAB measurements, {specimen_count} specimens; "
                f"Adams {scaling_label}"
            )
            axis.set_xlabel("Wavelength (nm)")
            axis.set_ylabel("Reflectance")
            axis.set_xlim(args.wavelength_min, args.wavelength_max)
            axis.grid(True, alpha=0.25)
            axis.legend(
                handles=[
                    Line2D(
                        [0],
                        [0],
                        color="#1f77b4",
                        linewidth=1.6,
                        label=f"All RELAB spectra for parent {parent}",
                    ),
                    Line2D(
                        [0],
                        [0],
                        color="#d62728",
                        linewidth=2.8,
                        label=f"Adams {adams_id} / {corrected_id}",
                    ),
                ],
                loc="best",
            )
            figure.tight_layout()
            figure.savefig(output_path, dpi=args.dpi)
            plt.close(figure)
            plots_written += 1

        specimen_ids = sorted(
            {canonical_id(row.get("relab_specimen_id")) for row in parent_relab}
        )
        measurement_ids = [
            canonical_id(row.get("measurement_id")) for row in parent_relab
        ]
        match_counts = defaultdict(int)
        for row in parent_relab:
            match_counts[clean(row.get("match_type"))] += 1

        plot_manifest.append(
            {
                "plot_number": plot_number,
                "plot_file": filename,
                "apollo_parent_id": parent,
                "atlas_join_status": clean(adams.get("atlas_join_status")),
                "relationship_scope": scope,
                "adams_specimen_id": adams_id,
                "adams_measurement_id": corrected_id,
                "adams_corrected_file": corrected_file,
                "coefficient_mode": args.coefficient,
                "coefficient_summary": (
                    "" if args.coefficient == "none" else str(paths["coefficient_summary"])
                ),
                "relab_measurement_count": len(parent_relab),
                "relab_specimen_count": len(specimen_ids),
                "relab_parent_record_count": match_counts["sample_number"],
                "relab_child_fraction_count": match_counts["subsample_or_fraction"],
                "relab_atlas_missing_count": match_counts["atlas_missing"],
                "relab_measurements_with_points": plotted_measurements,
                "relab_measurement_ids": ";".join(measurement_ids),
                "relab_specimen_ids": ";".join(specimen_ids),
                "missing_relab_point_ids": ";".join(missing_measurements),
            }
        )

        for relab in parent_relab:
            measurement = canonical_id(relab.get("measurement_id"))
            index = index_by_measurement.get(measurement, {})
            relab_manifest.append(
                {
                    "plot_number": plot_number,
                    "plot_file": filename,
                    "apollo_parent_id": parent,
                    "relationship_scope": scope,
                    "adams_specimen_id": adams_id,
                    "adams_measurement_id": corrected_id,
                    "relab_measurement_id": measurement,
                    "relab_specimen_id": clean(relab.get("relab_specimen_id")),
                    "relab_specimen_name": clean(relab.get("relab_specimen_name")),
                    "relab_match_type": clean(relab.get("match_type")),
                    "relab_needs_review": clean(relab.get("needs_review")),
                    "instrument_names": clean(index.get("instrument_names")),
                    "incidence_angle_deg": clean(index.get("incidence_angle_deg")),
                    "emission_angle_deg": clean(index.get("emission_angle_deg")),
                    "phase_angle_deg": clean(index.get("phase_angle_deg")),
                    "data_wavelength_min_nm": clean(index.get("data_wavelength_min_nm")),
                    "data_wavelength_max_nm": clean(index.get("data_wavelength_max_nm")),
                    "spectral_points_found": "yes" if measurement in relab_points else "no",
                }
            )

        print(
            f"[{plot_number:02d}/{len(selected_adams):02d}] parent {parent}, "
            f"Adams {adams_id}/{corrected_id}, RELAB {len(parent_relab)} -> {filename}"
        )

    plot_manifest_path = output_dir / "plot_manifest.csv"
    relab_manifest_path = output_dir / "plot_relab_records.csv"
    write_csv(plot_manifest_path, plot_manifest, list(plot_manifest[0]))
    write_csv(relab_manifest_path, relab_manifest, list(relab_manifest[0]))

    print(f"Adams plots selected: {len(selected_adams)}")
    print(f"PNG files written: {plots_written}")
    print(f"Existing PNG files skipped: {plots_skipped}")
    print(f"Distinct RELAB measurements loaded: {len(wanted_relab)}")
    print(f"Plot directory: {output_dir}")
    print(f"Plot manifest: {plot_manifest_path}")
    print(f"RELAB-record manifest: {relab_manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
