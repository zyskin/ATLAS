#!/usr/bin/env python3
"""Small end-to-end test using a generated two-page PDF."""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "preprocess_atlas_pdf.py"


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        pdf = root / "10003.pdf"
        out = root / "out"
        document = fitz.open()
        page = document.new_page()
        page.insert_text((72, 72), "Table 1a. Chemical composition of 10003")
        page.insert_text((72, 100), "reference Compston70 Rose70 Goles70 Rhodes80")
        page.insert_text((72, 128), "FeO 19.8 (c) 19.8 (d) 19.7 (e) 20.51 (c)")
        page.insert_text((72, 156), "technique: (c) XRF, (d) semi micro XRF, (e) INAA")
        page2 = document.new_page()
        page2.insert_text((72, 72), "Mineralogy includes plagioclase, pyroxene and ilmenite.")
        document.save(pdf)
        document.close()

        subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--pdf",
                str(pdf),
                "--document-id",
                "10003__test",
                "--sample-ids",
                "10003",
                "--source-url",
                "https://example.invalid/10003.pdf",
                "--output-dir",
                str(out),
            ],
            check=True,
        )
        with gzip.open(out / "json_gz" / "10003__test.json.gz", "rt", encoding="utf-8") as handle:
            parsed = json.load(handle)
        assert parsed["summary"]["page_count"] == 2
        assert parsed["summary"]["relevant_pages"] == 2
        assert parsed["pages"][0]["chemistry_layout"]["feo_rows"]
        assert parsed["pages"][0]["chemistry_layout"]["reference_rows"]
        assert (out / "ai_ready" / "10003__test.txt").is_file()
        assert (out / "status" / "10003__test.status.json").is_file()
    print("PASS: generated-PDF preprocessing test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
