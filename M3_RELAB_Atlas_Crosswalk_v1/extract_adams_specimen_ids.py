#!/usr/bin/env python3
"""List legacy LS-JBA specimen IDs contained in Adams_Spectra.txt."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_delimited(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        sample = handle.read(8192)
        handle.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        return list(csv.DictReader(handle, dialect=dialect))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(rows[0]) if rows else [
            "adams_specimen_id",
            "corrected_measurement_count",
            "corrected_measurement_ids",
            "corrected_filenames",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract unique LS-JBA specimen IDs and their corrected spectra."
    )
    parser.add_argument("--catalog", required=True, type=Path,
                        help="Adams_Spectra.txt")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--known-crosswalk", type=Path,
                        help="Optional validated crosswalk CSV; produces an unresolved list")
    args = parser.parse_args()

    grouped: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in read_delimited(args.catalog):
        specimen = (row.get("Sample ID") or "").strip().upper()
        filename = (row.get("Wavelength Corrected File") or "").strip()
        if not specimen:
            continue
        measurement = Path(filename).stem.upper() if filename else ""
        grouped[specimen].add((measurement, filename))

    rows: list[dict[str, str]] = []
    for specimen in sorted(grouped):
        pairs = sorted(grouped[specimen])
        rows.append({
            "adams_specimen_id": specimen,
            "corrected_measurement_count": str(len(pairs)),
            "corrected_measurement_ids": ";".join(p[0] for p in pairs),
            "corrected_filenames": ";".join(p[1] for p in pairs),
        })

    all_csv = args.output_dir / "adams_specimen_ids_all.csv"
    all_txt = args.output_dir / "adams_specimen_ids_all.txt"
    write_csv(all_csv, rows)
    all_txt.write_text(
        "\n".join(row["adams_specimen_id"] for row in rows) + "\n",
        encoding="utf-8",
    )

    print(f"Unique Adams specimen IDs: {len(rows)}")
    print(f"All IDs: {all_txt}")
    print(f"IDs with spectrum filenames: {all_csv}")

    if args.known_crosswalk:
        known = {
            (row.get("adams_specimen_id") or "").strip().upper()
            for row in read_delimited(args.known_crosswalk)
            if (row.get("adams_specimen_id") or "").strip()
        }
        unresolved = [row for row in rows if row["adams_specimen_id"] not in known]
        unresolved_csv = args.output_dir / "adams_specimen_ids_without_apollo_mapping.csv"
        unresolved_txt = args.output_dir / "adams_specimen_ids_without_apollo_mapping.txt"
        write_csv(unresolved_csv, unresolved)
        unresolved_txt.write_text(
            "\n".join(row["adams_specimen_id"] for row in unresolved) + "\n",
            encoding="utf-8",
        )
        print(f"Already represented in supplied crosswalk: {len(rows) - len(unresolved)}")
        print(f"Still without an Apollo mapping: {len(unresolved)}")
        print(f"Unresolved IDs: {unresolved_txt}")
        print(f"Unresolved IDs with spectrum filenames: {unresolved_csv}")


if __name__ == "__main__":
    main()
