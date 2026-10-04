#!/usr/bin/env python3
"""Offline test of cached-page discovery and TACC task-path replacement."""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "atlas_pdf_inventory.py"


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        atlas_csv = root / "atlas_samples.csv"
        cache = root / "cache"
        cache.mkdir()
        with atlas_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["atlas_sample_id", "mission", "detail_url"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "atlas_sample_id": "10003",
                    "mission": "Apollo 11",
                    "detail_url": "https://example.invalid/detail/10003",
                }
            )
        (cache / "10003.html").write_text(
            '<html><a href="../compendium/10003.pdf">Lunar Sample Compendium, 10003.pdf</a></html>',
            encoding="utf-8",
        )
        inventory = root / "inventory.csv"
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "discover",
                "--atlas-csv",
                str(atlas_csv),
                "--cache-dir",
                str(cache),
                "--output-csv",
                str(inventory),
            ],
            check=True,
        )
        with inventory.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1
        assert rows[0]["document_kind"] == "compendium"
        assert rows[0]["pdf_url"] == "https://example.invalid/compendium/10003.pdf"

        pdf_dir = root / "pdfs"
        pdf_dir.mkdir()
        document_id = "10003__compendium__abc"
        (pdf_dir / f"{document_id}.pdf").write_bytes(b"%PDF-test")
        manifest = root / "downloads.csv"
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "document_id",
                    "atlas_sample_ids",
                    "source_url",
                    "local_pdf",
                    "download_status",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "document_id": document_id,
                    "atlas_sample_ids": "10003",
                    "source_url": rows[0]["pdf_url"],
                    "local_pdf": "/wrong/machine/path.pdf",
                    "download_status": "downloaded",
                }
            )
        tasks = root / "tasks.tsv"
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "tasks",
                "--download-manifest",
                str(manifest),
                "--pdf-dir",
                str(pdf_dir),
                "--output-tasks",
                str(tasks),
            ],
            check=True,
        )
        assert str((pdf_dir / f"{document_id}.pdf").resolve()) in tasks.read_text(encoding="utf-8")
    print("PASS: offline Atlas inventory test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
