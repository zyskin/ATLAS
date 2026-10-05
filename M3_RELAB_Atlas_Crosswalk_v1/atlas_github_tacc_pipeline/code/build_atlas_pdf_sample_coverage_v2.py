#!/usr/bin/env python3
"""Find Atlas samples covered indirectly by already parsed Atlas PDFs.

The original PDF inventory records links found on each sample's Atlas detail
page.  A Lunar Sample Compendium PDF may cover several related samples even
when only one detail page links to it.  For example, 15030.pdf also discusses
15040 and its split 15041.

This program scans the full text stored in the Stage-1 ``json_gz`` files and
builds a conservative union of:

* direct associations: a sample detail page linked the PDF; and
* indirect associations: a known Atlas sample number occurs in the PDF text.

Indirect coverage is evidence that the document may be relevant.  It is NOT
permission to assign a chemistry value from one table column or parent sample
to the mentioned sample.  Every indirect association is therefore marked for
review.

Outputs
-------
atlas_pdf_sample_coverage_combined.csv
    One row per sample/document association, direct and indirect together.
atlas_indirect_pdf_snippets.csv
    Short, page-numbered contexts for every indirect association.
atlas_samples_without_pdf_coverage.csv
    Atlas rows not associated with any parsed PDF after the scan.
atlas_pdf_sample_coverage_v2_summary.json
    Counts, inputs, and explicit interpretation rules.

Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


VERSION = "atlas-pdf-sample-coverage-v2-1.0"

COMBINED_NAME = "atlas_pdf_sample_coverage_combined.csv"
SNIPPETS_NAME = "atlas_indirect_pdf_snippets.csv"
UNCOVERED_NAME = "atlas_samples_without_pdf_coverage.csv"
SUMMARY_NAME = "atlas_pdf_sample_coverage_v2_summary.json"

COMBINED_FIELDS = [
    "atlas_sample_id",
    "mission",
    "document_id",
    "source_url",
    "document_kind",
    "association_source",
    "relation_type",
    "association_needs_review",
    "direct_link_sample_ids",
    "mention_count",
    "mention_pages",
    "document_has_feo_rows",
    "document_feo_pages",
    "first_snippet",
]

SNIPPET_FIELDS = [
    "snippet_id",
    "atlas_sample_id",
    "document_id",
    "pdf_page",
    "relation_type",
    "source_url",
    "start_line",
    "end_line",
    "snippet",
]


def clean(value: object) -> str:
    return " ".join(str(value or "").split())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-csv", type=Path, required=True)
    parser.add_argument(
        "--parsed-dir",
        type=Path,
        required=True,
        help="Stage-1 parsed directory containing json_gz/, or json_gz itself",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--context-lines",
        type=int,
        default=2,
        help="Lines retained before and after a sample-number occurrence (default: 2)",
    )
    parser.add_argument(
        "--max-snippets-per-association",
        type=int,
        default=5,
        help="Bound output size by retaining at most this many contexts (default: 5)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No CSV header found in {path}")
        return [dict(row) for row in reader], list(reader.fieldnames)


def atlas_id_tokens(text: str) -> Iterable[str]:
    """Yield Atlas IDs, including 10072A and comma forms such as 15,041."""
    pattern = re.compile(
        r"(?<![A-Za-z0-9])"
        r"(?:(\d{5}[A-Za-z]+)|(\d{5})|(\d{2}),(\d{3})([A-Za-z]+)?)"
        r"(?![A-Za-z0-9])"
    )
    for match in pattern.finditer(text):
        if match.group(1):
            yield match.group(1)
        elif match.group(2):
            yield match.group(2)
        else:
            yield f"{match.group(3)}{match.group(4)}{match.group(5) or ''}"


def document_kind(url: str) -> str:
    lower = url.casefold()
    if "compendium" in lower:
        return "compendium"
    if "catalog" in lower or "catalogue" in lower:
        return "catalog"
    return "other_pdf"


def page_has_feo(page: dict[str, Any]) -> bool:
    layout = page.get("chemistry_layout") or {}
    if layout.get("feo_rows"):
        return True
    return bool(re.search(r"(?im)(?<![A-Za-z])FeO(?![A-Za-z])", str(page.get("text", ""))))


def classify_context(
    sample_id: str,
    page_number: int,
    line: str,
    context: str,
) -> tuple[str, int]:
    escaped = re.escape(sample_id)
    compact_context = " ".join(context.split())
    if re.search(
        rf"(?i)(?:chemical\s+composition[^\n]{{0,80}}{escaped}|"
        rf"{escaped}[^\n]{{0,80}}chemical\s+composition)",
        compact_context,
    ):
        return "chemistry_table_heading", 4
    if re.search(
        r"(?i)\b(?:processing|allocat(?:ion|ed|e)?|split|subsample|"
        r"daughter|parent|reserve|surface|subsurface)\b",
        compact_context,
    ):
        return "allocation_or_split", 3
    if re.fullmatch(rf"\s*{escaped}\s*", line) or (
        page_number == 1 and re.match(rf"\s*{escaped}\b", line)
    ):
        return "document_title_or_heading", 2
    if re.search(r"(?i)\b(?:sample|soil|rock|breccia|basalt|clast)\b", compact_context):
        return "sample_discussion", 1
    return "mentioned_in_pdf", 0


def load_document(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Top-level JSON is not an object: {path}")
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    temporary = path.with_name(path.name + ".part")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_document(
    document: dict[str, Any],
    known_ids: set[str],
    context_lines: int,
    max_snippets: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    source = document.get("source") or {}
    document_id = clean(source.get("document_id"))
    source_url = clean(source.get("source_url"))
    raw_direct = source.get("atlas_sample_ids") or []
    direct_ids = sorted({clean(item) for item in raw_direct if clean(item) in known_ids})
    warnings: list[str] = []
    if not document_id:
        raise ValueError("Parsed document is missing source.document_id")

    associations: dict[str, dict[str, Any]] = {}
    for sample_id in direct_ids:
        associations[sample_id] = {
            "count": 0,
            "pages": set(),
            "relation_type": "direct_detail_page_link",
            "relation_rank": 10,
            "snippets": [],
        }

    feo_pages: list[int] = []
    for page in document.get("pages") or []:
        page_number = int(page.get("page_number") or 0)
        text = str(page.get("text") or "")
        if page_has_feo(page):
            feo_pages.append(page_number)
        lines = text.splitlines()
        for line_index, line in enumerate(lines):
            ids_on_line = [item for item in atlas_id_tokens(line) if item in known_ids]
            if not ids_on_line:
                continue
            start = max(0, line_index - context_lines)
            end = min(len(lines), line_index + context_lines + 1)
            context = "\n".join(lines[start:end]).strip()
            for sample_id in ids_on_line:
                item = associations.setdefault(
                    sample_id,
                    {
                        "count": 0,
                        "pages": set(),
                        "relation_type": "mentioned_in_pdf",
                        "relation_rank": 0,
                        "snippets": [],
                    },
                )
                item["count"] += 1
                item["pages"].add(page_number)
                if sample_id not in direct_ids:
                    relation, rank = classify_context(sample_id, page_number, line, context)
                    if rank > item["relation_rank"]:
                        item["relation_type"] = relation
                        item["relation_rank"] = rank
                    signature = (page_number, start + 1, end, context)
                    if signature not in item["snippets"] and len(item["snippets"]) < max_snippets:
                        item["snippets"].append(signature)

    combined_rows: list[dict[str, Any]] = []
    snippet_rows: list[dict[str, Any]] = []
    for sample_id in sorted(associations):
        item = associations[sample_id]
        indirect = sample_id not in direct_ids
        first_snippet = ""
        if item["snippets"]:
            page_number, start, end, context = item["snippets"][0]
            first_snippet = f"p{page_number} lines {start}-{end}: {clean(context)}"
        combined_rows.append(
            {
                "atlas_sample_id": sample_id,
                "document_id": document_id,
                "source_url": source_url,
                "document_kind": document_kind(source_url),
                "association_source": "pdf_text_indirect" if indirect else "direct_detail_page_link",
                "relation_type": item["relation_type"],
                "association_needs_review": "yes" if indirect else "no",
                "direct_link_sample_ids": ",".join(direct_ids),
                "mention_count": item["count"],
                "mention_pages": ",".join(str(value) for value in sorted(item["pages"])),
                "document_has_feo_rows": "yes" if feo_pages else "no",
                "document_feo_pages": ",".join(str(value) for value in sorted(set(feo_pages))),
                "first_snippet": first_snippet,
            }
        )
        if indirect:
            for page_number, start, end, context in item["snippets"]:
                snippet_rows.append(
                    {
                        "atlas_sample_id": sample_id,
                        "document_id": document_id,
                        "pdf_page": page_number,
                        "relation_type": item["relation_type"],
                        "source_url": source_url,
                        "start_line": start,
                        "end_line": end,
                        "snippet": context,
                    }
                )

    unknown_direct = sorted({clean(item) for item in raw_direct if clean(item)} - known_ids)
    if unknown_direct:
        warnings.append(
            f"{document_id}: direct IDs absent from atlas CSV: {','.join(unknown_direct)}"
        )
    return combined_rows, snippet_rows, warnings


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.context_lines < 0 or args.max_snippets_per_association < 1:
        print("ERROR: context lines must be >=0 and max snippets must be >=1", file=sys.stderr)
        return 2
    json_dir = args.parsed_dir / "json_gz" if (args.parsed_dir / "json_gz").is_dir() else args.parsed_dir
    files = sorted(json_dir.glob("*.json.gz"))
    if not files:
        print(f"ERROR: no *.json.gz files found in {json_dir}", file=sys.stderr)
        return 2

    outputs = [
        args.output_dir / COMBINED_NAME,
        args.output_dir / SNIPPETS_NAME,
        args.output_dir / UNCOVERED_NAME,
        args.output_dir / SUMMARY_NAME,
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        print("ERROR: output exists; add --overwrite to replace it:", file=sys.stderr)
        for path in existing:
            print(f"  {path}", file=sys.stderr)
        return 2

    try:
        atlas_rows, atlas_fields = read_csv(args.atlas_csv)
        if "atlas_sample_id" not in atlas_fields:
            raise ValueError(f"{args.atlas_csv} is missing atlas_sample_id")
        atlas_by_id: dict[str, dict[str, str]] = {}
        for row_number, row in enumerate(atlas_rows, 2):
            sample_id = clean(row.get("atlas_sample_id"))
            # The Atlas contains ordinary five-digit sample numbers and a
            # small number of explicitly indexed letter-suffixed records,
            # currently including 10072A and 10072D.
            if not re.fullmatch(r"\d{5}[A-Za-z]*", sample_id):
                raise ValueError(f"Invalid Atlas sample ID {sample_id!r} at CSV row {row_number}")
            atlas_by_id.setdefault(sample_id, row)
        known_ids = set(atlas_by_id)

        combined: list[dict[str, Any]] = []
        snippets: list[dict[str, Any]] = []
        warnings: list[str] = []
        errors: list[str] = []
        for number, path in enumerate(files, 1):
            try:
                rows, snippet_rows, document_warnings = scan_document(
                    load_document(path),
                    known_ids,
                    args.context_lines,
                    args.max_snippets_per_association,
                )
                combined.extend(rows)
                snippets.extend(snippet_rows)
                warnings.extend(document_warnings)
            except (OSError, EOFError, gzip.BadGzipFile, json.JSONDecodeError, ValueError) as exc:
                errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
            if number % 100 == 0:
                print(f"Scanned {number}/{len(files)} parsed PDFs", file=sys.stderr)

        mission_by_id = {
            sample_id: clean(row.get("mission")) for sample_id, row in atlas_by_id.items()
        }
        for row in combined:
            row["mission"] = mission_by_id.get(str(row["atlas_sample_id"]), "")
        combined.sort(key=lambda row: (str(row["atlas_sample_id"]), str(row["document_id"])))
        snippets.sort(
            key=lambda row: (
                str(row["atlas_sample_id"]), str(row["document_id"]), int(row["pdf_page"]),
                int(row["start_line"]),
            )
        )
        for number, row in enumerate(snippets, 1):
            row["snippet_id"] = number

        covered_ids = {str(row["atlas_sample_id"]) for row in combined}
        uncovered_fields = list(atlas_fields)
        if "coverage_status" not in uncovered_fields:
            uncovered_fields.append("coverage_status")
        uncovered: list[dict[str, Any]] = []
        for sample_id in sorted(known_ids):
            if sample_id not in covered_ids:
                uncovered.append(
                    {**atlas_by_id[sample_id], "coverage_status": "no_direct_or_indirect_parsed_pdf"}
                )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(outputs[0], combined, COMBINED_FIELDS)
        write_csv(outputs[1], snippets, SNIPPET_FIELDS)
        write_csv(outputs[2], uncovered, uncovered_fields)

        direct_count = sum(row["association_source"] == "direct_detail_page_link" for row in combined)
        indirect_count = len(combined) - direct_count
        indirect_ids = {
            str(row["atlas_sample_id"])
            for row in combined
            if row["association_source"] == "pdf_text_indirect"
        }
        summary = {
            "schema": "atlas-pdf-sample-coverage-v2-summary-1.0",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "program": VERSION,
            "inputs": {
                "atlas_csv": str(args.atlas_csv),
                "parsed_json_dir": str(json_dir),
                "parsed_pdf_count": len(files),
            },
            "counts": {
                "atlas_sample_count": len(known_ids),
                "sample_document_associations": len(combined),
                "direct_associations": direct_count,
                "indirect_associations": indirect_count,
                "distinct_indirect_sample_ids": len(indirect_ids),
                "indirect_snippets": len(snippets),
                "samples_without_parsed_pdf_coverage": len(uncovered),
                "warnings": len(warnings),
                "errors": len(errors),
            },
            "outputs": {},
            "interpretation_rules": [
                "An indirect association means only that a known Atlas sample ID occurs in the PDF text.",
                "Every indirect association requires review.",
                "A FeO row elsewhere in the same document is not automatically assigned to an indirectly mentioned sample.",
                "Table-column ownership and parent/child transfer must be established separately.",
            ],
            "warnings": warnings,
            "errors": errors,
        }
        for path in outputs[:3]:
            summary["outputs"][path.name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        outputs[3].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, csv.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Parsed PDFs scanned: {len(files)}")
    print(f"Combined direct/indirect associations: {len(combined)}")
    print(f"Indirect associations needing review: {indirect_count}")
    print(f"Distinct indirectly discovered samples: {len(indirect_ids)}")
    print(f"Samples still without parsed-PDF coverage: {len(uncovered)}")
    print(f"Combined table: {outputs[0]}")
    print(f"Indirect snippets: {outputs[1]}")
    print(f"Uncovered samples: {outputs[2]}")
    print(f"Summary: {outputs[3]}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
