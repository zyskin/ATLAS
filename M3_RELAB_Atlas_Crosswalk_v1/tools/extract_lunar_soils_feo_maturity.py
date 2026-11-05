#!/usr/bin/env python3
"""Extract FeO (wt.%) and Is/FeO from JSC 19069, Handbook of Lunar Soils.

The handbook is a scanned/OCR PDF.  This program uses Poppler's
``pdftotext -layout`` output because preserving the table rows is important.
It finds each data-sheet heading (``SOIL: nnnnn``), searches that sheet and
up to two following pages, and writes:

* lunar_soils_feo_maturity.csv       - the requested three-column table
* lunar_soils_feo_maturity_audit.csv - pages, evidence, corrections, status
* lunar_soils_feo_maturity_summary.json

Missing source cells remain empty.  The program does not estimate them.

Ubuntu dependency::

    sudo apt install poppler-utils

Example::

    python extract_lunar_soils_feo_maturity.py LunarSoils.pdf \
        --output-dir lunar_soils_extracted
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


HEADING_RE = re.compile(r"^\s*SOIL\s*:\s*([0-9]{4,6})\b", re.IGNORECASE)
FEO_RE = re.compile(r"^\s*Fe[O0]\s+([0-9]+(?:\.[0-9]+)?)\b", re.IGNORECASE)
IS_FEO_RE = re.compile(
    r"I\s*[sS]?\s*/\s*Fe[O0]\s*:?[\s,]*([0-9]+(?:\.[0-9]+)?)\b",
    re.IGNORECASE,
)

# Confirmed printed heading typo in the handbook: viewer page 79 describes
# Apollo 12 soil 12003 but prints SOIL: 21003.
SAMPLE_ID_CORRECTIONS = {
    "21003": ("12003", "Printed handbook heading typo: SOIL 21003 corrected to 12003"),
}


@dataclass
class Match:
    value: float
    page: int
    evidence: str


@dataclass
class Record:
    sample_id: str
    raw_sample_id: str
    heading_page: int
    feo_wt_percent: float | None
    feo_page: int | None
    is_feo: float | None
    is_feo_page: int | None
    status: str
    heading_text: str
    feo_evidence: str
    is_feo_evidence: str
    correction_note: str


def extract_layout_text(pdf: Path, pdftotext: str) -> str:
    executable = shutil.which(pdftotext)
    if not executable:
        raise RuntimeError(
            f"{pdftotext!r} was not found. On Ubuntu install poppler-utils."
        )
    with tempfile.TemporaryDirectory(prefix="lunar_soils_") as temp_dir:
        output = Path(temp_dir) / "layout.txt"
        subprocess.run(
            [executable, "-layout", str(pdf), str(output)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return output.read_text(encoding="utf-8", errors="replace")


def find_headings(pages: list[str]) -> list[tuple[int, str, str]]:
    headings: list[tuple[int, str, str]] = []
    for page_number, page_text in enumerate(pages, start=1):
        for line in page_text.splitlines():
            match = HEADING_RE.match(line)
            if match:
                headings.append((page_number, match.group(1), line.strip()))
    return headings


def find_matches(
    pages: list[str], start_page: int, end_page: int, pattern: re.Pattern[str]
) -> list[Match]:
    matches: list[Match] = []
    for page_number in range(start_page, end_page + 1):
        for line in pages[page_number - 1].splitlines():
            match = pattern.search(line)
            if match:
                matches.append(
                    Match(float(match.group(1)), page_number, " ".join(line.split()))
                )
    return matches


def one_or_none(matches: list[Match]) -> Match | None:
    return matches[0] if len(matches) == 1 else None


def status_for(feo_matches: list[Match], maturity_matches: list[Match]) -> str:
    problems: list[str] = []
    if not feo_matches:
        problems.append("missing_FeO")
    elif len(feo_matches) > 1:
        problems.append("multiple_FeO_values")
    if not maturity_matches:
        problems.append("missing_Is_FeO")
    elif len(maturity_matches) > 1:
        problems.append("multiple_Is_FeO_values")
    return ";".join(problems) if problems else "complete"


def parse(text: str, max_pages_per_sheet: int) -> list[Record]:
    pages = text.split("\f")
    if pages and not pages[-1].strip():
        pages.pop()

    headings = find_headings(pages)
    if not headings:
        raise RuntimeError("No lines matching 'SOIL: sample-number' were found.")

    records: list[Record] = []
    for index, (heading_page, raw_id, heading_text) in enumerate(headings):
        next_heading_page = (
            headings[index + 1][0] if index + 1 < len(headings) else len(pages) + 1
        )
        # Some data sheets continue onto a second page.  The cap prevents the
        # last sheet in a mission from absorbing the following bibliography.
        end_page = min(
            next_heading_page - 1,
            heading_page + max_pages_per_sheet - 1,
            len(pages),
        )

        feo_matches = find_matches(pages, heading_page, end_page, FEO_RE)
        maturity_matches = find_matches(pages, heading_page, end_page, IS_FEO_RE)
        feo = one_or_none(feo_matches)
        maturity = one_or_none(maturity_matches)

        if raw_id in SAMPLE_ID_CORRECTIONS:
            sample_id, correction_note = SAMPLE_ID_CORRECTIONS[raw_id]
        else:
            sample_id, correction_note = raw_id, ""

        records.append(
            Record(
                sample_id=sample_id,
                raw_sample_id=raw_id,
                heading_page=heading_page,
                feo_wt_percent=feo.value if feo else None,
                feo_page=feo.page if feo else None,
                is_feo=maturity.value if maturity else None,
                is_feo_page=maturity.page if maturity else None,
                status=status_for(feo_matches, maturity_matches),
                heading_text=heading_text,
                feo_evidence=" || ".join(m.evidence for m in feo_matches),
                is_feo_evidence=" || ".join(m.evidence for m in maturity_matches),
                correction_note=correction_note,
            )
        )
    return records


def write_outputs(records: list[Record], output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    simple_path = output_dir / "lunar_soils_feo_maturity.csv"
    audit_path = output_dir / "lunar_soils_feo_maturity_audit.csv"
    summary_path = output_dir / "lunar_soils_feo_maturity_summary.json"

    with simple_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "FeO_wt_percent", "Is_FeO"])
        for record in records:
            writer.writerow(
                [
                    record.sample_id,
                    "" if record.feo_wt_percent is None else record.feo_wt_percent,
                    "" if record.is_feo is None else record.is_feo,
                ]
            )

    audit_fields = list(asdict(records[0]).keys())
    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_fields)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)

    both = sum(r.feo_wt_percent is not None and r.is_feo is not None for r in records)
    summary = {
        "schema": "lunar-soils-feo-maturity-1.0",
        "records": len(records),
        "FeO_values_extracted": sum(r.feo_wt_percent is not None for r in records),
        "Is_FeO_values_extracted": sum(r.is_feo is not None for r in records),
        "records_with_both_values": both,
        "records_needing_review_or_with_missing_values": sum(
            r.status != "complete" for r in records
        ),
        "known_heading_corrections_applied": sum(bool(r.correction_note) for r in records),
        "missing_values_are_blank_and_were_not_estimated": True,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return simple_path, audit_path, summary_path


def validate_known_example(records: list[Record]) -> None:
    record = next((r for r in records if r.sample_id == "14260"), None)
    if record is None:
        raise RuntimeError("Validation failed: sample 14260 was not found.")
    if record.feo_wt_percent != 11.08 or record.is_feo != 72.0:
        raise RuntimeError(
            "Validation failed for 14260: expected FeO=11.08 and Is/FeO=72.0; "
            f"found FeO={record.feo_wt_percent!r}, Is/FeO={record.is_feo!r}."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Handbook of Lunar Soils PDF")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("lunar_soils_extracted")
    )
    parser.add_argument(
        "--pdftotext",
        default="pdftotext",
        help="pdftotext executable name or path (default: pdftotext)",
    )
    parser.add_argument(
        "--max-pages-per-sheet",
        type=int,
        default=3,
        help="maximum pages searched after each SOIL heading (default: 3)",
    )
    args = parser.parse_args()

    if not args.pdf.is_file():
        parser.error(f"PDF does not exist: {args.pdf}")
    if args.max_pages_per_sheet < 1:
        parser.error("--max-pages-per-sheet must be at least 1")

    text = extract_layout_text(args.pdf, args.pdftotext)
    records = parse(text, args.max_pages_per_sheet)
    validate_known_example(records)
    simple_path, audit_path, summary_path = write_outputs(records, args.output_dir)

    print(f"Soil sheets found: {len(records)}")
    print(f"FeO values extracted: {sum(r.feo_wt_percent is not None for r in records)}")
    print(f"Is/FeO values extracted: {sum(r.is_feo is not None for r in records)}")
    print(
        "Both values extracted: "
        f"{sum(r.feo_wt_percent is not None and r.is_feo is not None for r in records)}"
    )
    print("Validation: 14260 -> FeO 11.08 wt.%, Is/FeO 72.0 (PASS)")
    print(f"Table: {simple_path}")
    print(f"Audit: {audit_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
