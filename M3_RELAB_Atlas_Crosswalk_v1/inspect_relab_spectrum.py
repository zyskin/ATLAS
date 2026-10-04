#!/usr/bin/env python3
"""Inspect one RELAB spectrum, its FeO association and descriptive metadata.

The native wavelength grid is plotted without interpolation, smoothing or
normalization.  A new plotting segment is started whenever wavelength fails to
increase, so detector/measurement resets are not joined by a misleading line.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path


def clean(value: object) -> str:
    return " ".join(str(value or "").split())


def parse_optional_float(value: str) -> float | None:
    text = clean(value)
    return float(text) if text else None


def find_index_row(path: Path, measurement_id: str) -> dict[str, str] | None:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if clean(row.get("measurement_id", "")) == measurement_id:
                return dict(row)
    return None


def find_table_selection(
    path: Path, table_id: int
) -> tuple[dict[str, str] | None, list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    selected = next((row for row in rows if clean(row.get("id", "")) == str(table_id)), None)
    if selected is None:
        return None, []
    measurement_id = clean(selected.get("relab_measurement_id", ""))
    related = [
        row for row in rows if clean(row.get("relab_measurement_id", "")) == measurement_id
    ]
    return selected, related


def find_spectral_points(path: Path, measurement_id: str) -> list[dict[str, object]]:
    points: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"measurement_id", "point_index", "wavelength_nm", "reflectance"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing spectral-point columns: {', '.join(sorted(missing))}")
        for row in reader:
            if clean(row.get("measurement_id", "")) != measurement_id:
                continue
            points.append(
                {
                    "point_index": int(row["point_index"]),
                    "wavelength_nm": float(row["wavelength_nm"]),
                    "reflectance": float(row["reflectance"]),
                    "uncertainty": parse_optional_float(row.get("uncertainty_value", "")),
                    "uncertainty_type": clean(row.get("uncertainty_type", "")),
                }
            )
    points.sort(key=lambda point: int(point["point_index"]))
    return points


def unique_nonempty(values):
    result = []
    seen = set()
    for value in values:
        value = clean(value)
        if value and value.casefold() not in seen:
            result.append(value)
            seen.add(value.casefold())
    return result


def format_information(
    table_id: int,
    measurement_id: str,
    selected_row: dict[str, str],
    index_row: dict[str, str] | None,
    table_rows: list[dict[str, str]],
    points: list[dict[str, object]],
) -> str:
    lines: list[str] = []
    lines.append(f"Our table ID: {table_id}")
    lines.append(f"Apollo parent ID: {clean(selected_row.get('i_atlas_sample', '')) or 'unknown'}")
    lines.append(f"Child/subsample: {clean(selected_row.get('j_child', '')) or 'unknown'}")
    lines.append(f"RELAB measurement ID: {measurement_id}")
    lines.append(
        "RELAB specimen ID: "
        + (clean((index_row or {}).get("specimen_id", "")) or "unknown")
    )
    lines.append("=" * max(20, 32 + len(measurement_id)))

    lines.append("\nFeO content")
    lines.append("-----------")
    feo_records = []
    seen_feo = set()
    selected_has_feo = clean(selected_row.get("feo_known", "")).casefold() == "yes"
    # Put the chemistry record represented by the selected table row first.
    ordered_rows = [selected_row] + [row for row in table_rows if row is not selected_row]
    for row in ordered_rows:
        if clean(row.get("feo_known", "")).casefold() != "yes":
            continue
        key = (
            clean(row.get("k_feo_wt_percent", "")),
            clean(row.get("chemistry_product_id", "")),
            clean(row.get("chemistry_specimen_id", "")),
            clean(row.get("chemistry_phase", "")),
            clean(row.get("chemistry_source_and_method", "")),
            clean(row.get("feo_association", "")),
        )
        if key not in seen_feo:
            feo_records.append((row is selected_row, key))
            seen_feo.add(key)

    if not feo_records:
        lines.append("Unknown: no FeO association was found in the extracted RELAB chemistry.")
    else:
        if not selected_has_feo:
            lines.append("Selected-row FeO: unknown (-99)")
        for is_selected, record in feo_records:
            feo, product, specimen, phase, source, association = record
            prefix = "Selected-row FeO" if is_selected else "Other associated FeO"
            lines.append(f"{prefix}: {feo} wt.%")
            if phase:
                lines.append(f"  Chemistry phase: {phase}")
            if specimen:
                lines.append(f"  Chemistry specimen: {specimen}")
            if product:
                lines.append(f"  Chemistry product: {product}")
            if association:
                lines.append(f"  Association: {association}")
            if source:
                lines.append(f"  Source/method: {source}")

    lines.append("\nSample and material information")
    lines.append("-------------------------------")
    first_table = selected_row
    fields = [
        ("RELAB specimen name", (index_row or {}).get("specimen_name", "")),
        ("Description", (index_row or {}).get("specimen_description", "")),
        ("Collection location", (index_row or {}).get("specimen_collection_location", "")),
        ("Specimen type", (index_row or {}).get("specimen_type", "")),
        ("Material origin", (index_row or {}).get("material_origin", "")),
        ("Material type", (index_row or {}).get("material_type", "")),
        ("Material subtypes", (index_row or {}).get("material_subtypes", "")),
        ("Atlas sample type", first_table.get("atlas_sample_type", "")),
        ("Atlas lithology", first_table.get("atlas_lithology", "")),
    ]
    wrote_field = False
    for label, value in fields:
        value = clean(value)
        if value:
            lines.append(f"{label}: {value}")
            wrote_field = True
    if not wrote_field:
        lines.append("No sample-description metadata found.")

    comments = unique_nonempty(row.get("comments", "") for row in table_rows)
    treatment = unique_nonempty(row.get("age_or_treatment_notes", "") for row in table_rows)
    if treatment:
        lines.append("\nAge, maturity or treatment notes")
        lines.append("--------------------------------")
        lines.extend(treatment)
    if comments:
        lines.append("\nSubstantial comments")
        lines.append("--------------------")
        lines.extend(comments)

    lines.append("\nMeasurement information")
    lines.append("-----------------------")
    measurement_fields = [
        ("Instrument", (index_row or {}).get("instrument_names", "")),
        ("Measurement segments", (index_row or {}).get("measurement_segments", "")),
        ("Incidence angle", (index_row or {}).get("incidence_angle_deg", "")),
        ("Emission angle", (index_row or {}).get("emission_angle_deg", "")),
        ("Phase angle", (index_row or {}).get("phase_angle_deg", "")),
        ("Native fields", (index_row or {}).get("native_field_names", "")),
    ]
    for label, value in measurement_fields:
        value = clean(value)
        if value:
            suffix = " degrees" if "angle" in label.casefold() else ""
            lines.append(f"{label}: {value}{suffix}")

    if points:
        wavelengths = [float(point["wavelength_nm"]) for point in points]
        nonincreasing = sum(
            wavelengths[n] <= wavelengths[n - 1] for n in range(1, len(wavelengths))
        )
        lines.append(f"Spectral points: {len(points)}")
        lines.append(f"Wavelength range: {min(wavelengths):g} to {max(wavelengths):g} nm")
        lines.append(f"Non-increasing wavelength transitions: {nonincreasing}")
        uncertainty_types = unique_nonempty(
            point["uncertainty_type"] for point in points if point["uncertainty"] is not None
        )
        lines.append(
            "Uncertainty: " + (", ".join(uncertainty_types) if uncertainty_types else "not supplied")
        )
    else:
        lines.append("No spectral points found.")

    urls = unique_nonempty(
        value
        for row in table_rows
        for value in (row.get("atlas_detail_url", ""), row.get("atlas_pdf_url", ""))
    )
    if urls:
        lines.append("\nAtlas sources")
        lines.append("-------------")
        lines.extend(urls)

    return "\n".join(lines) + "\n"


def split_monotone_segments(points: list[dict[str, object]]):
    segments = []
    current = []
    previous = None
    for point in points:
        wavelength = float(point["wavelength_nm"])
        if previous is not None and wavelength <= previous and current:
            segments.append(current)
            current = []
        current.append(point)
        previous = wavelength
    if current:
        segments.append(current)
    return segments


def plot_spectrum(
    points: list[dict[str, object]],
    title: str,
    output_png: Path,
    xmin: float | None,
    xmax: float | None,
    save: bool,
    show: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Run this in your gdal environment "
            "or install matplotlib in the active environment."
        ) from exc

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = False
    uncertainty_plotted = False
    for segment in split_monotone_segments(points):
        selected = [
            point
            for point in segment
            if (xmin is None or float(point["wavelength_nm"]) >= xmin)
            and (xmax is None or float(point["wavelength_nm"]) <= xmax)
        ]
        if not selected:
            continue
        x = [float(point["wavelength_nm"]) for point in selected]
        y = [float(point["reflectance"]) for point in selected]
        ax.plot(x, y, color="black", linewidth=1.4, label="Reflectance" if not plotted else None)
        plotted = True

        if all(point["uncertainty"] is not None for point in selected):
            uncertainty = [float(point["uncertainty"]) for point in selected]
            lower = [value - error for value, error in zip(y, uncertainty)]
            upper = [value + error for value, error in zip(y, uncertainty)]
            ax.fill_between(
                x,
                lower,
                upper,
                color="tab:blue",
                alpha=0.20,
                linewidth=0,
                label="Reported uncertainty" if not uncertainty_plotted else None,
            )
            uncertainty_plotted = True

    if not plotted:
        raise ValueError("No spectral points fall inside the requested wavelength interval")

    ax.set_title(title)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    if save:
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=180)
        print(f"Plot saved: {output_png}")
    if show:
        plt.show()
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Enter our sequential table ID; display its native RELAB spectrum and "
            "all available extracted information."
        )
    )
    parser.add_argument("table_id", nargs="?", type=int, help="Our sequential ID, e.g. 1243")
    parser.add_argument("--spectral-points", type=Path, required=True)
    parser.add_argument("--spectrum-index", type=Path, required=True)
    parser.add_argument("--atlas-feo-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("relab_spectrum_plots"))
    parser.add_argument("--xmin", type=float, help="Optional minimum plotted wavelength in nm")
    parser.add_argument("--xmax", type=float, help="Optional maximum plotted wavelength in nm")
    parser.add_argument("--no-save", action="store_true", help="Show only; do not save PNG or TXT")
    parser.add_argument("--no-show", action="store_true", help="Save only; do not open plot window")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    table_id = args.table_id
    if table_id is None:
        entered = input("Our table ID: ").strip()
        try:
            table_id = int(entered)
        except ValueError:
            print(f"ERROR: table ID must be an integer, got {entered!r}", file=sys.stderr)
            return 2
    if table_id < 1:
        print("ERROR: table ID must be a positive integer", file=sys.stderr)
        return 2
    if args.xmin is not None and args.xmax is not None and args.xmin >= args.xmax:
        print("ERROR: --xmin must be less than --xmax", file=sys.stderr)
        return 2
    for path in (args.spectral_points, args.spectrum_index, args.atlas_feo_table):
        if not path.is_file():
            print(f"ERROR: input file not found: {path}", file=sys.stderr)
            return 2

    try:
        selected_row, table_rows = find_table_selection(args.atlas_feo_table, table_id)
        if selected_row is None:
            raise ValueError(f"Our table ID was not found: {table_id}")
        measurement_id = clean(selected_row.get("relab_measurement_id", ""))
        if not measurement_id:
            raise ValueError(f"Table ID {table_id} has no RELAB measurement ID")
        index_row = find_index_row(args.spectrum_index, measurement_id)
        points = find_spectral_points(args.spectral_points, measurement_id)

        information = format_information(
            table_id, measurement_id, selected_row, index_row, table_rows, points
        )
        print("\n" + information)

        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", measurement_id)
        output_png = args.output_dir / f"Atlas_ID_{table_id}_RELAB_{safe_id}_spectrum.png"
        output_txt = args.output_dir / f"Atlas_ID_{table_id}_RELAB_{safe_id}_information.txt"
        if not args.no_save:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            output_txt.write_text(information, encoding="utf-8")
            print(f"Information saved: {output_txt}")

        if not points:
            raise ValueError(f"No spectral points found for {measurement_id}")
        specimen_name = clean((index_row or {}).get("specimen_name", ""))
        title = f"ID {table_id}: {measurement_id}" + (
            f" — {specimen_name}" if specimen_name else ""
        )
        plot_spectrum(
            points,
            title,
            output_png,
            args.xmin,
            args.xmax,
            save=not args.no_save,
            show=not args.no_show,
        )
    except (OSError, ValueError, csv.Error, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
