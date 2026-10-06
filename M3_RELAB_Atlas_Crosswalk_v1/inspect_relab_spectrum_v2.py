#!/usr/bin/env python3
"""Display one numbered RELAB record as native curve, M3 points, or both.

This is a new version of inspect_relab_spectrum.py.  It imports the original
metadata/FeO formatting functions and does not replace or modify the old tool.
At the end it reports automated Atlas-PDF extraction and the separate,
traceable AI-aided PDF/publication search tables when matching records exist.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

from inspect_relab_spectrum import (
    clean,
    find_index_row,
    find_spectral_points,
    find_table_selection,
    format_information,
    split_monotone_segments,
)


BASE = Path(__file__).resolve().parent
RAW = BASE / "relab_raw_native_2026-10-04"
STRUCTURED = BASE / "data" / "structured_v1"
AI_DIR = BASE / "data" / "ai_resolution_v1"


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def sample_tokens(value: object) -> set[str]:
    return set(re.findall(r"(?<!\d)\d{5}(?!\d)", clean(value)))


def find_m3_points(path: Path, measurement_id: str) -> list[dict[str, float | int]]:
    result = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {
            "measurement_id", "m3_band", "m3_response_centroid_nm",
            "m3_reflectance", "bandpass_coverage_fraction", "usable",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing M3-table columns: {', '.join(sorted(missing))}")
        for row in reader:
            if clean(row.get("measurement_id")) != measurement_id:
                continue
            if clean(row.get("usable")).casefold() != "yes" or not clean(row.get("m3_reflectance")):
                continue
            result.append(
                {
                    "band": int(row["m3_band"]),
                    "wavelength_nm": float(row["m3_response_centroid_nm"]),
                    "reflectance": float(row["m3_reflectance"]),
                    "coverage": float(row["bandpass_coverage_fraction"]),
                }
            )
    result.sort(key=lambda row: int(row["band"]))
    return result


def format_pdf_extraction(path: Path, parent_id: str, limit: int = 20) -> str:
    lines = [
        "Automated Atlas PDF extraction",
        "------------------------------",
        "This is separate from RELAB metadata and has not necessarily been AI-adjudicated.",
    ]
    matches = [
        row for row in read_rows(path)
        if parent_id in sample_tokens(row.get("atlas_sample_id", ""))
    ]
    if not matches:
        lines.append(f"No automated PDF FeO candidate was found for Apollo parent {parent_id}.")
        return "\n".join(lines)
    matches.sort(key=lambda row: clean(row.get("needs_review")).casefold() == "yes")
    for row in matches[:limit]:
        value = clean(row.get("value_wt_percent")) or "unknown"
        qualifier = clean(row.get("qualifier"))
        uncertainty = clean(row.get("uncertainty_wt_percent"))
        lines.append(
            f"FeO: {qualifier}{value} wt.%"
            + (f" ± {uncertainty} wt.%" if uncertainty else "")
        )
        lines.append(
            "  Status: "
            + ("NEEDS REVIEW" if clean(row.get("needs_review")).casefold() == "yes" else "parser accepted")
        )
        for label, field in (
            ("Scope", "sample_scope"), ("Reference", "reference"),
            ("Method", "analytical_method"), ("PDF page", "pdf_page"),
            ("Evidence", "evidence_text"), ("Source", "source_url"),
            ("Review reason", "review_reason"),
        ):
            value_text = clean(row.get(field))
            if value_text:
                lines.append(f"  {label}: {value_text}")
    if len(matches) > limit:
        lines.append(f"{len(matches) - limit} additional PDF candidates omitted from display.")
    return "\n".join(lines)


def format_ai_findings(directory: Path, parent_id: str) -> str:
    lines = [
        "AI-aided search/review of PDFs and publications",
        "------------------------------------------------",
        "This section is explicitly separate from RELAB metadata.",
    ]
    found = False

    adjudications = read_rows(directory / "review_candidate_adjudications.csv")
    relevant = [
        row for row in adjudications
        if parent_id in sample_tokens(row.get("candidate_parent_id", ""))
        or parent_id in sample_tokens(row.get("assigned_sample_id", ""))
    ]
    accepted = [row for row in relevant if clean(row.get("decision")).startswith("accepted")]
    for row in accepted:
        found = True
        lines.append(
            f"Accepted PDF FeO: {clean(row.get('value_wt_percent')) or 'unknown'} wt.%"
        )
        for label, field in (
            ("Assigned sample", "assigned_sample_id"), ("Scope", "sample_scope"),
            ("Reference", "reference"), ("Method", "analytical_method"),
            ("PDF page", "pdf_page"), ("Evidence", "evidence_text"),
            ("Reason", "adjudication_reason"), ("Source", "source_url"),
        ):
            value = clean(row.get(field))
            if value:
                lines.append(f"  {label}: {value}")
    rejected_count = sum(clean(row.get("decision")).startswith("rejected") for row in relevant)
    if rejected_count:
        lines.append(f"Rejected parser false positives for this parent: {rejected_count}")

    for row in read_rows(directory / "missing_case_resolutions.csv"):
        if parent_id not in sample_tokens(row.get("apollo_parent_id", "")):
            continue
        found = True
        lines.append(f"AI/search resolution status: {clean(row.get('resolution_status'))}")
        accepted_feo = clean(row.get("accepted_feo_wt_percent"))
        retained = clean(row.get("retained_nonbulk_feo_wt_percent"))
        if accepted_feo:
            lines.append(f"  Accepted FeO: {accepted_feo} wt.%")
        if retained:
            lines.append(f"  Retained non-bulk/ambiguous FeO: {retained} wt.%")
        for label, field in (
            ("Scope", "accepted_scope"), ("Evidence", "evidence_summary"),
            ("Review note", "review_note"), ("Primary source", "primary_source_url"),
            ("Secondary source", "secondary_source_url"),
        ):
            value = clean(row.get(field))
            if value:
                lines.append(f"  {label}: {value}")

    for filename in (
        "incidental_al2o3_age_maturity.csv",
        "missing_case_incidental_al2o3_age_maturity.csv",
    ):
        for row in read_rows(directory / filename):
            if parent_id not in sample_tokens(row.get("sample_id", "")):
                continue
            found = True
            value = clean(row.get("value_text"))
            unit = clean(row.get("unit"))
            lines.append(
                f"Other useful PDF finding: {clean(row.get('property'))}: {value}"
                + (f" {unit}" if unit else "")
            )
            for label, field in (
                ("Scope", "scope"), ("PDF page", "pdf_page"),
                ("Evidence", "evidence"), ("Source", "source_url"),
            ):
                text = clean(row.get(field))
                if text:
                    lines.append(f"  {label}: {text}")

    if not found:
        lines.append(
            f"No separate AI-aided PDF/publication finding is recorded for Apollo parent {parent_id}."
        )
    return "\n".join(lines)


def plot_selected(
    native_points,
    m3_points,
    view: str,
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
        raise RuntimeError("matplotlib is required for plotting") from exc

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = False
    if view in {"relab", "both"}:
        first = True
        for segment in split_monotone_segments(native_points):
            selected = [
                point for point in segment
                if (xmin is None or float(point["wavelength_nm"]) >= xmin)
                and (xmax is None or float(point["wavelength_nm"]) <= xmax)
            ]
            if not selected:
                continue
            ax.plot(
                [float(point["wavelength_nm"]) for point in selected],
                [float(point["reflectance"]) for point in selected],
                color="black", linewidth=1.25,
                label="Native RELAB curve" if first else None,
                zorder=2,
            )
            first = False
            plotted = True
    if view in {"m3", "both"}:
        selected_m3 = [
            point for point in m3_points
            if (xmin is None or float(point["wavelength_nm"]) >= xmin)
            and (xmax is None or float(point["wavelength_nm"]) <= xmax)
        ]
        if selected_m3:
            ax.plot(
                [float(point["wavelength_nm"]) for point in selected_m3],
                [float(point["reflectance"]) for point in selected_m3],
                linestyle="None", marker="o", markersize=6.2,
                markeredgewidth=1.4, color="tab:blue",
                label="Forward-modelled M3 band values", zorder=5,
            )
            plotted = True
    if not plotted:
        raise ValueError("No requested spectral values fall inside the plotted interval")

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table_id", nargs="?", type=int, help="Our sequential ID, e.g. 1243")
    parser.add_argument("--view", choices=("relab", "m3", "both"), default="both")
    parser.add_argument("--spectral-points", type=Path, default=RAW / "relab_spectral_points.csv")
    parser.add_argument("--spectrum-index", type=Path, default=RAW / "relab_spectrum_index.csv")
    parser.add_argument("--atlas-feo-table", type=Path, default=RAW / "atlas_relab_feo_table.csv")
    parser.add_argument(
        "--m3-spectra", type=Path,
        default=BASE / "data" / "relab_m3_v1" / "relab_m3_spectra.csv",
    )
    parser.add_argument(
        "--pdf-feo-table", type=Path,
        default=STRUCTURED / "atlas_feo_measurements.csv",
    )
    parser.add_argument("--ai-resolution-dir", type=Path, default=AI_DIR)
    parser.add_argument("--output-dir", type=Path, default=BASE / "relab_spectrum_plots_v2")
    parser.add_argument("--xmin", type=float)
    parser.add_argument("--xmax", type=float)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    table_id = args.table_id
    if table_id is None:
        try:
            table_id = int(input("Our table ID: ").strip())
        except ValueError:
            print("ERROR: table ID must be an integer", file=sys.stderr)
            return 2
    if table_id < 1:
        print("ERROR: table ID must be positive", file=sys.stderr)
        return 2
    if args.xmin is not None and args.xmax is not None and args.xmin >= args.xmax:
        print("ERROR: --xmin must be less than --xmax", file=sys.stderr)
        return 2
    required = [args.spectral_points, args.spectrum_index, args.atlas_feo_table]
    if args.view in {"m3", "both"}:
        required.append(args.m3_spectra)
    for path in required:
        if not path.is_file():
            print(f"ERROR: input file not found: {path}", file=sys.stderr)
            return 2

    try:
        selected_row, table_rows = find_table_selection(args.atlas_feo_table, table_id)
        if selected_row is None:
            raise ValueError(f"Our table ID was not found: {table_id}")
        measurement_id = clean(selected_row.get("relab_measurement_id"))
        if not measurement_id:
            raise ValueError(f"Table ID {table_id} has no RELAB measurement ID")
        parent_id = clean(selected_row.get("i_atlas_sample"))
        index_row = find_index_row(args.spectrum_index, measurement_id)
        native_points = find_spectral_points(args.spectral_points, measurement_id)
        m3_points = (
            find_m3_points(args.m3_spectra, measurement_id)
            if args.view in {"m3", "both"} else []
        )

        information = format_information(
            table_id, measurement_id, selected_row, index_row, table_rows, native_points
        ).rstrip()
        information += "\n\nM3 forward-model summary\n------------------------\n"
        if args.view == "relab":
            information += "M3 table was not read because --view relab was selected.\n"
        elif m3_points:
            information += (
                f"Usable M3 bands: {len(m3_points)}\n"
                f"Band range: {m3_points[0]['band']} to {m3_points[-1]['band']}\n"
                "Displayed as independent blue points; no curve is fitted through them.\n"
            )
        else:
            information += "No M3 band met the recorded coverage requirement.\n"
        information += "\n" + format_pdf_extraction(args.pdf_feo_table, parent_id)
        information += "\n\n" + format_ai_findings(args.ai_resolution_dir, parent_id) + "\n"
        print("\n" + information)

        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", measurement_id)
        stem = f"Atlas_ID_{table_id}_RELAB_{safe_id}_{args.view}"
        output_png = args.output_dir / f"{stem}.png"
        output_txt = args.output_dir / f"{stem}_information.txt"
        if not args.no_save:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            output_txt.write_text(information, encoding="utf-8")
            print(f"Information saved: {output_txt}")

        specimen_name = clean((index_row or {}).get("specimen_name"))
        title = f"ID {table_id}: {measurement_id}" + (
            f" — {specimen_name}" if specimen_name else ""
        )
        plot_selected(
            native_points, m3_points, args.view, title, output_png,
            args.xmin, args.xmax, save=not args.no_save, show=not args.no_show,
        )
        return 0
    except (OSError, ValueError, csv.Error, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
