#!/usr/bin/env python3
"""Convert one Lunar Sample Atlas PDF into compact, traceable AI-ready data.

No scientific interpretation and no AI call occurs here.  The output preserves:
  * source URL and SHA-256 of the exact PDF;
  * page numbers and full extracted text;
  * keyword-selected evidence with line numbers;
  * word coordinates on relevant pages, needed to reconstruct table columns;
  * FeO/reference/technique layout rows for later deterministic table parsing.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF


PARSER_VERSION = "atlas-pdf-preprocessor-1.0"

PATTERNS = {
    "chemistry": re.compile(
        r"(?i)\b(?:FeO|TiO2|Al2O3|MgO|CaO|Na2O|K2O|MnO|SiO2|"
        r"chemical\s+composition|bulk\s+composition|major\s+elements?|"
        r"trace\s+elements?|wt\.?\s*%|weight\s+percent)\b"
    ),
    "mineralogy": re.compile(
        r"(?i)\b(?:mineralogy|mineralogical|modal|mode|plagioclase|pyroxene|"
        r"clinopyroxene|orthopyroxene|olivine|ilmenite|spinel|anorthite|"
        r"troctolite|norite|glass|agglutinate)\b"
    ),
    "material": re.compile(
        r"(?i)\b(?:soil|regolith|rock|basalt|breccia|impact\s+melt|rake|core|"
        r"fragment|clast|matrix|particulate|powder|grain\s+size)\b"
    ),
    "age": re.compile(
        r"(?i)\b(?:age|dated|dating|crystallization|exposure\s+age|cosmic[- ]ray|"
        r"\d+(?:\.\d+)?\s*(?:Ga|Ma|Gyr|Myr))\b"
    ),
    "treatment": re.compile(
        r"(?i)\b(?:irradiat(?:ed|ion)|heated|annealed|weathered|mature|maturity|"
        r"altered|separated|sieved|fraction|magnetic(?:ally)?|hand[- ]picked)\b"
    ),
    "references": re.compile(r"(?i)\b(?:references?|bibliography|technique|method)\b"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--sample-ids", default="")
    parser.add_argument("--source-url", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-lines", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_write_reproducible_gzip_json(path: Path, value: Any) -> None:
    """Write deterministic gzip bytes (mtime=0) with stable JSON ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            zipped.write(payload)
    temporary.replace(path)


def clean_text(value: str) -> str:
    return "\n".join(line.rstrip() for line in value.replace("\x00", "").splitlines()).strip()


def merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def evidence_blocks(lines: list[str], context: int) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for category, pattern in PATTERNS.items():
        hits = [index for index, line in enumerate(lines) if pattern.search(line)]
        ranges = merge_ranges(
            [(max(0, index - context), min(len(lines), index + context + 1)) for index in hits]
        )
        for start, end in ranges:
            blocks.append(
                {
                    "category": category,
                    "start_line": start + 1,
                    "end_line": end,
                    "text": "\n".join(lines[start:end]).strip(),
                }
            )
    return blocks


def group_word_rows(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for word in words:
        grouped.setdefault((word["block"], word["line"]), []).append(word)
    rows = []
    for (block, line), members in grouped.items():
        members.sort(key=lambda item: (item["x0"], item["word"]))
        text = " ".join(item["text"] for item in members)
        rows.append(
            {
                "block": block,
                "line": line,
                "y0": min(item["y0"] for item in members),
                "y1": max(item["y1"] for item in members),
                "text": text,
                "words": members,
            }
        )
    rows.sort(key=lambda item: (item["y0"], item["block"], item["line"]))
    return rows


def chemistry_layout(rows: list[dict[str, Any]]) -> dict[str, Any]:
    feo_rows = [row for row in rows if re.search(r"(?i)(?<![A-Za-z])FeO(?![A-Za-z])", row["text"])]
    reference_rows = [row for row in rows if re.search(r"(?i)^\s*reference\b", row["text"])]
    technique_rows = [row for row in rows if re.search(r"(?i)\b(?:technique|method)\s*:", row["text"])]
    return {
        "feo_rows": feo_rows,
        "reference_rows": reference_rows,
        "technique_rows": technique_rows,
    }


def extract_document(args: argparse.Namespace, input_sha: str) -> tuple[dict[str, Any], str]:
    document = fitz.open(args.pdf)
    pages: list[dict[str, Any]] = []
    ai_parts = [
        f"DOCUMENT_ID: {args.document_id}",
        f"ATLAS_SAMPLE_IDS: {args.sample_ids}",
        f"SOURCE_URL: {args.source_url}",
        f"SOURCE_PDF_SHA256: {input_sha}",
        f"PARSER: {PARSER_VERSION}",
        "",
    ]
    pages_with_text = 0
    relevant_pages = 0
    evidence_count = 0

    for page_number, page in enumerate(document, 1):
        text = clean_text(page.get_text("text", sort=True))
        if text:
            pages_with_text += 1
        lines = text.splitlines()
        blocks = evidence_blocks(lines, args.context_lines)
        categories = sorted({block["category"] for block in blocks})
        evidence_count += len(blocks)

        page_record: dict[str, Any] = {
            "page_number": page_number,
            "width_points": round(float(page.rect.width), 3),
            "height_points": round(float(page.rect.height), 3),
            "text": text,
            "text_sha256": sha256_bytes(text.encode("utf-8")),
            "categories": categories,
            "evidence_blocks": blocks,
            "word_coordinates": [],
            "chemistry_layout": {"feo_rows": [], "reference_rows": [], "technique_rows": []},
        }

        if blocks:
            relevant_pages += 1
            raw_words = page.get_text("words", sort=False)
            words = [
                {
                    "x0": round(float(word[0]), 3),
                    "y0": round(float(word[1]), 3),
                    "x1": round(float(word[2]), 3),
                    "y1": round(float(word[3]), 3),
                    "text": str(word[4]),
                    "block": int(word[5]),
                    "line": int(word[6]),
                    "word": int(word[7]),
                }
                for word in raw_words
            ]
            page_record["word_coordinates"] = words
            page_record["chemistry_layout"] = chemistry_layout(group_word_rows(words))

            ai_parts.append(f"=== PDF PAGE {page_number}; CATEGORIES: {','.join(categories)} ===")
            for block_number, block in enumerate(blocks, 1):
                ai_parts.append(
                    f"--- {block['category']} block {block_number}; "
                    f"lines {block['start_line']}-{block['end_line']} ---"
                )
                ai_parts.append(block["text"])
            ai_parts.append("")
        pages.append(page_record)

    document.close()
    parsed = {
        "schema": "lunar-sample-atlas-preprocessed-1.0",
        "parser": {"name": PARSER_VERSION, "pymupdf_version": fitz.__version__},
        "source": {
            "document_id": args.document_id,
            "atlas_sample_ids": [item for item in args.sample_ids.split(",") if item],
            "source_url": args.source_url,
            "source_filename": args.pdf.name,
            "source_pdf_sha256": input_sha,
            "source_pdf_size_bytes": args.pdf.stat().st_size,
        },
        "summary": {
            "page_count": len(pages),
            "pages_with_extractable_text": pages_with_text,
            "relevant_pages": relevant_pages,
            "evidence_block_count": evidence_count,
            "needs_ocr": pages_with_text == 0,
        },
        "pages": pages,
    }
    return parsed, "\n".join(ai_parts).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    args.pdf = args.pdf.expanduser().resolve()
    if not args.pdf.is_file():
        print(f"ERROR: PDF not found: {args.pdf}", file=sys.stderr)
        return 2

    json_dir = args.output_dir / "json_gz"
    ai_dir = args.output_dir / "ai_ready"
    status_dir = args.output_dir / "status"
    json_path = json_dir / f"{args.document_id}.json.gz"
    ai_path = ai_dir / f"{args.document_id}.txt"
    status_path = status_dir / f"{args.document_id}.status.json"
    input_sha = sha256_file(args.pdf)

    if status_path.is_file() and json_path.is_file() and ai_path.is_file() and not args.overwrite:
        try:
            old = json.loads(status_path.read_text(encoding="utf-8"))
            if (
                old.get("status") == "ok"
                and old.get("source_pdf_sha256") == input_sha
                and old.get("parser_version") == PARSER_VERSION
            ):
                print(f"Already complete: {args.document_id}")
                return 0
        except (json.JSONDecodeError, OSError):
            pass

    status: dict[str, Any] = {
        "schema": "lunar-sample-atlas-preprocess-status-1.0",
        "document_id": args.document_id,
        "atlas_sample_ids": args.sample_ids,
        "source_url": args.source_url,
        "source_pdf_filename": args.pdf.name,
        "source_pdf_sha256": input_sha,
        "parser_version": PARSER_VERSION,
        "processed_utc": utc_now(),
    }
    try:
        parsed, ai_text = extract_document(args, input_sha)
        atomic_write_reproducible_gzip_json(json_path, parsed)
        atomic_write_text(ai_path, ai_text)
        status.update(
            {
                "status": "ok",
                "page_count": parsed["summary"]["page_count"],
                "pages_with_extractable_text": parsed["summary"]["pages_with_extractable_text"],
                "relevant_pages": parsed["summary"]["relevant_pages"],
                "evidence_block_count": parsed["summary"]["evidence_block_count"],
                "needs_ocr": parsed["summary"]["needs_ocr"],
                "json_gz": f"json_gz/{json_path.name}",
                "json_gz_sha256": sha256_file(json_path),
                "ai_ready": f"ai_ready/{ai_path.name}",
                "ai_ready_sha256": sha256_file(ai_path),
                "error": "",
            }
        )
        atomic_write_text(status_path, json.dumps(status, indent=2, sort_keys=True) + "\n")
        print(
            f"Processed {args.document_id}: {status['page_count']} pages, "
            f"{status['evidence_block_count']} evidence blocks"
        )
        return 0
    except Exception as exc:
        status.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        atomic_write_text(status_path, json.dumps(status, indent=2, sort_keys=True) + "\n")
        print(f"ERROR processing {args.document_id}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
