#!/usr/bin/env python3
"""Collect the LPI Lunar Sample Atlas index into a reproducible CSV file.

The program can either download the live index or parse an HTML file saved
from a browser.  The latter is useful if the LPI site presents an anti-bot
page to command-line programs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


DEFAULT_URL = "https://www.lpi.usra.edu/lunar/samples/atlas/index.cfm"
SAMPLE_RE = re.compile(r"^[0-9]{5}(?:[A-Za-z]+)?$")


class AtlasTableParser(HTMLParser):
    """Small HTML table reader; keeps cell text, header status, and first link."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.table_depth = 0
        self.tables: list[list[list[dict[str, object]]]] = []
        self.current_table: list[list[dict[str, object]]] | None = None
        self.current_row: list[dict[str, object]] | None = None
        self.current_cell: dict[str, object] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attributes = dict(attrs)
        if tag == "table":
            if self.table_depth == 0:
                self.current_table = []
            self.table_depth += 1
        elif tag == "tr" and self.table_depth == 1:
            self.current_row = []
        elif tag in {"th", "td"} and self.table_depth == 1 and self.current_row is not None:
            self.current_cell = {"parts": [], "is_header": tag == "th", "href": ""}
        elif tag == "a" and self.current_cell is not None and not self.current_cell["href"]:
            self.current_cell["href"] = attributes.get("href") or ""

    def handle_data(self, data: str) -> None:
        if self.current_cell is not None:
            parts = self.current_cell["parts"]
            assert isinstance(parts, list)
            parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"th", "td"} and self.current_cell is not None and self.current_row is not None:
            parts = self.current_cell.pop("parts")
            assert isinstance(parts, list)
            self.current_cell["text"] = clean_text(" ".join(parts))
            self.current_row.append(self.current_cell)
            self.current_cell = None
        elif tag == "tr" and self.table_depth == 1 and self.current_row is not None:
            if self.current_table is not None and self.current_row:
                self.current_table.append(self.current_row)
            self.current_row = None
        elif tag == "table" and self.table_depth:
            self.table_depth -= 1
            if self.table_depth == 0 and self.current_table is not None:
                self.tables.append(self.current_table)
                self.current_table = None


def clean_text(value: str) -> str:
    return " ".join(value.replace("\xa0", " ").split())


def normalized_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", clean_text(value).lower()).strip()


def infer_mission(sample_id: str) -> str:
    """Infer the Apollo mission from the standard sample-number prefix."""
    digits = "".join(re.findall(r"\d", sample_id))
    if len(digits) < 2:
        return ""
    return {
        "10": "Apollo 11",
        "12": "Apollo 12",
        "14": "Apollo 14",
        "15": "Apollo 15",
        "60": "Apollo 16",
        "61": "Apollo 16",
        "62": "Apollo 16",
        "63": "Apollo 16",
        "64": "Apollo 16",
        "65": "Apollo 16",
        "66": "Apollo 16",
        "67": "Apollo 16",
        "68": "Apollo 16",
        "69": "Apollo 16",
        "70": "Apollo 17",
        "71": "Apollo 17",
        "72": "Apollo 17",
        "73": "Apollo 17",
        "74": "Apollo 17",
        "75": "Apollo 17",
        "76": "Apollo 17",
        "77": "Apollo 17",
        "78": "Apollo 17",
        "79": "Apollo 17",
    }.get(digits[:2], "")


def find_column(headers: list[str], candidates: tuple[str, ...]) -> int | None:
    for index, header in enumerate(headers):
        if header in candidates:
            return index
    for index, header in enumerate(headers):
        if any(candidate in header for candidate in candidates):
            return index
    return None


def parse_atlas_html(html: str, base_url: str = DEFAULT_URL) -> list[dict[str, str]]:
    parser = AtlasTableParser()
    parser.feed(html)
    parser.close()
    results: list[dict[str, str]] = []

    for rows in parser.tables:
        if not rows:
            continue

        header_row_index = None
        headers: list[str] = []
        for row_index, row in enumerate(rows[:10]):
            candidate = [normalized_header(str(cell.get("text", ""))) for cell in row]
            joined = " | ".join(candidate)
            if "sample" in joined and "description" in joined:
                headers = candidate
                header_row_index = row_index
                break

        if header_row_index is None:
            continue

        sample_col = find_column(headers, ("sample", "sample number", "sample id"))
        type_col = find_column(headers, ("sample type", "type"))
        lithology_col = find_column(headers, ("lithology",))
        description_col = find_column(headers, ("description",))
        if sample_col is None or description_col is None:
            continue

        for row in rows[header_row_index + 1 :]:
            if sample_col >= len(row):
                continue
            sample_id = clean_text(str(row[sample_col].get("text", "")))
            if not SAMPLE_RE.fullmatch(sample_id):
                continue

            href = str(row[sample_col].get("href", ""))
            detail_url = urljoin(base_url, href) if href else ""

            def cell_text(index: int | None) -> str:
                if index is None or index >= len(row):
                    return ""
                return clean_text(str(row[index].get("text", "")))

            results.append(
                {
                    "atlas_sample_id": sample_id,
                    "mission": infer_mission(sample_id),
                    "sample_type": cell_text(type_col),
                    "lithology": cell_text(lithology_col),
                    "description": cell_text(description_col),
                    "detail_url": detail_url,
                    "atlas_pdf_url": "",
                    "source_index_url": base_url,
                }
            )

    if not results:
        page_text = clean_text(re.sub(r"<[^>]+>", " ", html)).lower()
        if "cloudflare" in page_text or "just a moment" in page_text:
            raise ValueError(
                "The input is an anti-bot page, not the Atlas index. Open the Atlas "
                "in a browser, save the complete HTML page, and rerun with --input-html."
            )
        raise ValueError("No Lunar Sample Atlas table was found in the supplied HTML.")

    # Preserve the first occurrence while making duplicate handling explicit.
    unique: dict[str, dict[str, str]] = {}
    for record in results:
        unique.setdefault(record["atlas_sample_id"], record)
    return sorted(unique.values(), key=lambda row: row["atlas_sample_id"])


def download_html(url: str, timeout: float) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "Chrome/140 Safari/537.36 M3-Lunar-Research-Catalogue/1.0"
        )
    }
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(charset, errors="replace")
    except HTTPError as exc:
        if exc.code == 403:
            raise RuntimeError(
                "The Atlas returned HTTP 403. Save the Atlas index from a browser and "
                "use --input-html /path/to/saved_page.html."
            ) from exc
        raise


def write_csv(records: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "atlas_sample_id",
        "mission",
        "sample_type",
        "lithology",
        "description",
        "detail_url",
        "atlas_pdf_url",
        "source_index_url",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input-html", type=Path, help="Atlas HTML saved from a browser")
    source.add_argument("--url", default=DEFAULT_URL, help="Atlas index URL")
    parser.add_argument("--output", type=Path, default=Path("atlas_samples.csv"))
    parser.add_argument("--metadata", type=Path, help="Optional metadata JSON path")
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.input_html:
            raw = args.input_html.read_bytes()
            html = raw.decode("utf-8", errors="replace")
            source_description = str(args.input_html.resolve())
            base_url = DEFAULT_URL
        else:
            html = download_html(args.url, args.timeout)
            raw = html.encode("utf-8")
            source_description = args.url
            base_url = args.url

        records = parse_atlas_html(html, base_url=base_url)
        write_csv(records, args.output)

        metadata_path = args.metadata or args.output.with_suffix(".metadata.json")
        metadata = {
            "schema": "lpi-lunar-sample-atlas-index-1.0",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source": source_description,
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "record_count": len(records),
            "output_csv": str(args.output.resolve()),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"Atlas catalogue written: {args.output} ({len(records)} unique samples)")
        print(f"Provenance metadata written: {metadata_path}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
