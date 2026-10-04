#!/usr/bin/env python3
"""End-to-end test: HTTP download, extraction, and temporary-PDF deletion."""

from __future__ import annotations

import http.server
import os
import socketserver
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "fetch_preprocess_atlas_pdf.py"


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        served = root / "served"
        temp_downloads = root / "tmp"
        output = root / "output"
        served.mkdir()
        temp_downloads.mkdir()

        pdf = served / "10003.pdf"
        document = fitz.open()
        page = document.new_page()
        page.insert_text((72, 72), "FeO 19.8 wt.% by XRF")
        document.save(pdf)
        document.close()

        handler = lambda *a, **kw: QuietHandler(*a, directory=str(served), **kw)
        with socketserver.TCPServer(("127.0.0.1", 0), handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            url = f"http://127.0.0.1:{server.server_address[1]}/10003.pdf"
            environment = dict(os.environ)
            environment["TMPDIR"] = str(temp_downloads)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--url",
                    url,
                    "--document-id",
                    "10003__fetch_test",
                    "--sample-ids",
                    "10003",
                    "--output-dir",
                    str(output),
                ],
                env=environment,
                check=False,
            )
            server.shutdown()
            thread.join()

        assert result.returncode == 0
        assert not list(temp_downloads.glob("*.pdf")), "temporary PDF was not deleted"
        assert (output / "json_gz" / "10003__fetch_test.json.gz").is_file()
        assert (output / "ai_ready" / "10003__fetch_test.txt").is_file()
        assert (output / "status" / "10003__fetch_test.status.json").is_file()
    print("PASS: temporary PDF downloaded, processed, and deleted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
