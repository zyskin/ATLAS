#!/usr/bin/env python3
"""Download one Atlas PDF temporarily, preprocess it, and always delete it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from preprocess_atlas_pdf import PARSER_VERSION


USER_AGENT = "M3-Lunar-Groundtruth/1.0 (research; one PDF per task)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--sample-ids", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-source-change",
        action="store_true",
        help="Allow a URL to replace a previously recorded, different PDF hash.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_error_status(
    args: argparse.Namespace,
    status: str,
    error: str,
    previous_sha256: str = "",
    downloaded_sha256: str = "",
) -> None:
    path = args.output_dir / "status" / f"{args.document_id}.status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {
        "schema": "lunar-sample-atlas-preprocess-status-1.0",
        "document_id": args.document_id,
        "atlas_sample_ids": args.sample_ids,
        "source_url": args.url,
        "parser_version": PARSER_VERSION,
        "processed_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "error": error,
        "source_pdf_sha256": previous_sha256,
        "downloaded_pdf_sha256": downloaded_sha256,
    }
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    status_path = args.output_dir / "status" / f"{args.document_id}.status.json"
    previous: dict[str, object] = {}
    if status_path.is_file():
        try:
            previous = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
    if (
        not args.overwrite
        and previous.get("status") == "ok"
        and previous.get("parser_version") == PARSER_VERSION
    ):
        print(f"Already complete; no download needed: {args.document_id}")
        return 0

    temp_root = Path(os.environ.get("TMPDIR", tempfile.gettempdir()))
    temp_root.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f"{args.document_id}.", suffix=".pdf", dir=temp_root, delete=False
    )
    temp_pdf = Path(handle.name)
    handle.close()
    try:
        request = Request(args.url, headers={"User-Agent": USER_AGENT})
        with urlopen(request, timeout=args.timeout) as response, temp_pdf.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        with temp_pdf.open("rb") as check:
            if check.read(5) != b"%PDF-":
                raise ValueError("downloaded response does not begin with %PDF-")

        current_sha = sha256_file(temp_pdf)
        previous_sha = str(previous.get("source_pdf_sha256", ""))
        if previous_sha and current_sha != previous_sha and not args.allow_source_change:
            message = (
                f"source URL content changed: previous SHA-256 {previous_sha}, "
                f"current SHA-256 {current_sha}"
            )
            write_error_status(
                args,
                "source_hash_changed",
                message,
                previous_sha256=previous_sha,
                downloaded_sha256=current_sha,
            )
            print(f"ERROR: {message}", file=sys.stderr)
            return 3

        command = [
            sys.executable,
            str(Path(__file__).with_name("preprocess_atlas_pdf.py")),
            "--pdf",
            str(temp_pdf),
            "--document-id",
            args.document_id,
            "--sample-ids",
            args.sample_ids,
            "--source-url",
            args.url,
            "--output-dir",
            str(args.output_dir),
        ]
        if args.overwrite:
            command.append("--overwrite")
        return subprocess.run(command, check=False).returncode
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        write_error_status(args, "download_error", message)
        print(f"ERROR downloading {args.document_id}: {message}", file=sys.stderr)
        return 2
    finally:
        try:
            temp_pdf.unlink(missing_ok=True)
        except OSError as exc:
            print(f"WARNING: could not delete temporary PDF {temp_pdf}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
