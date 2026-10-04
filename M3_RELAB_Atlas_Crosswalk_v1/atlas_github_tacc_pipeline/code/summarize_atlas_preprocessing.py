#!/usr/bin/env python3
"""Combine per-PDF status files into a CSV and a compact JSON summary."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


FIELDS = [
    "document_id",
    "atlas_sample_ids",
    "status",
    "page_count",
    "pages_with_extractable_text",
    "relevant_pages",
    "evidence_block_count",
    "needs_ocr",
    "source_url",
    "source_pdf_sha256",
    "json_gz",
    "json_gz_sha256",
    "ai_ready",
    "ai_ready_sha256",
    "parser_version",
    "processed_utc",
    "error",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--index-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for path in sorted((args.output_dir / "status").glob("*.status.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            row = {"document_id": path.stem, "status": "unreadable_status", "error": str(exc)}
        rows.append({field: row.get(field, "") for field in FIELDS})

    args.index_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.index_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(str(row["status"]) for row in rows)
    summary = {
        "schema": "lunar-sample-atlas-preprocess-summary-1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status_file_count": len(rows),
        "status_counts": dict(sorted(counts.items())),
        "needs_ocr_count": sum(str(row["needs_ocr"]).lower() == "true" for row in rows),
        "total_pages": sum(int(row["page_count"] or 0) for row in rows),
        "total_evidence_blocks": sum(int(row["evidence_block_count"] or 0) for row in rows),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
