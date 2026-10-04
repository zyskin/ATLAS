#!/usr/bin/env python3
"""Discover and download Lunar Sample Atlas PDFs reproducibly.

The Atlas index contains thousands of sample-detail pages, but only some detail
pages link to a Lunar Sample Compendium or another PDF.  This program keeps the
three stages separate and resumable:

  discover  Visit/cache each detail page and record every PDF link.
  download  Download each unique PDF once and record its SHA-256 digest.
  tasks     Write a tab-separated task file for the TACC Slurm array.

Only Python's standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import Request, urlopen


USER_AGENT = "M3-Lunar-Groundtruth/1.0 (research; one request at a time)"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return value or "document"


class AnchorParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.current: dict[str, object] | None = None
        self.anchors: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "a":
            self.current = {"href": dict(attrs).get("href") or "", "parts": []}

    def handle_data(self, data: str) -> None:
        if self.current is not None:
            parts = self.current["parts"]
            assert isinstance(parts, list)
            parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self.current is not None:
            href = str(self.current["href"])
            parts = self.current["parts"]
            assert isinstance(parts, list)
            text = " ".join(" ".join(parts).split())
            self.anchors.append((href, text))
            self.current = None


def classify_link(text: str, url: str) -> str:
    value = f"{text} {url}".lower()
    if "compendium" in value:
        return "compendium"
    if "catalog" in value or "catalogue" in value:
        return "catalog"
    return "other_pdf"


def fetch(url: str, timeout: float) -> tuple[bytes, str]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        final_url = response.geturl()
        return response.read(), final_url


def read_atlas_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"atlas_sample_id", "detail_url"}
    missing = required.difference(rows[0] if rows else {})
    if missing:
        raise ValueError(f"Atlas CSV is missing columns: {sorted(missing)}")
    return rows


DISCOVERY_FIELDS = [
    "atlas_sample_id",
    "mission",
    "detail_url",
    "detail_sha256",
    "detail_checked_utc",
    "discovery_status",
    "link_text",
    "pdf_url",
    "document_kind",
    "error",
]


def command_discover(args: argparse.Namespace) -> int:
    rows = read_atlas_rows(args.atlas_csv)
    if args.cache_dir is not None:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.start_at:
        rows = [row for row in rows if row["atlas_sample_id"] >= args.start_at]
    if args.limit:
        rows = rows[: args.limit]

    output_rows: list[dict[str, str]] = []
    processed_sample_ids: set[str] = set()
    if args.output_csv.is_file() and not args.restart:
        with args.output_csv.open(newline="", encoding="utf-8-sig") as handle:
            existing = list(csv.DictReader(handle))
        if existing and set(DISCOVERY_FIELDS).difference(existing[0]):
            raise ValueError(
                "Existing discovery CSV has an incompatible schema; use --restart "
                "or choose another output path"
            )
        output_rows.extend(existing)
        processed_sample_ids = {
            row.get("atlas_sample_id", "") for row in existing if row.get("atlas_sample_id", "")
        }
        rows = [row for row in rows if row["atlas_sample_id"] not in processed_sample_ids]
        if processed_sample_ids:
            print(
                f"Resuming: {len(processed_sample_ids)} sample detail pages already recorded",
                file=sys.stderr,
            )

    def save_checkpoint() -> None:
        temporary = args.output_csv.with_name(args.output_csv.name + ".part")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=DISCOVERY_FIELDS)
            writer.writeheader()
            writer.writerows(output_rows)
        temporary.replace(args.output_csv)

    live_requests = 0
    for number, row in enumerate(rows, 1):
        sample_id = row["atlas_sample_id"].strip()
        detail_url = row["detail_url"].strip()
        cache_path = (
            args.cache_dir / f"{safe_name(sample_id)}.html"
            if args.cache_dir is not None
            else None
        )
        checked = utc_now()
        error = ""
        status = ""
        content = b""

        try:
            if cache_path is not None and cache_path.is_file() and not args.refresh:
                content = cache_path.read_bytes()
                status = "cached"
            elif not detail_url:
                status = "missing_detail_url"
            else:
                if live_requests and args.delay:
                    time.sleep(args.delay)
                content, _ = fetch(detail_url, args.timeout)
                live_requests += 1
                if cache_path is not None:
                    cache_path.write_bytes(content)
                status = "downloaded"
        except Exception as exc:  # keep a complete audit row and continue
            status = "error"
            error = f"{type(exc).__name__}: {exc}"

        links: list[tuple[str, str, str]] = []
        if content:
            parser = AnchorParser()
            parser.feed(content.decode("utf-8", errors="replace"))
            parser.close()
            for href, text in parser.anchors:
                absolute = urljoin(detail_url, href)
                parsed = urlparse(absolute)
                if parsed.path.lower().endswith(".pdf"):
                    links.append((text, absolute, classify_link(text, absolute)))

        common = {
            "atlas_sample_id": sample_id,
            "mission": row.get("mission", ""),
            "detail_url": detail_url,
            "detail_sha256": sha256_bytes(content) if content else "",
            "detail_checked_utc": checked,
            "discovery_status": status if links else f"{status}_no_pdf",
            "error": error,
        }
        if links:
            for text, pdf_url, kind in links:
                output_rows.append(
                    {**common, "link_text": text, "pdf_url": pdf_url, "document_kind": kind}
                )
        else:
            output_rows.append(
                {**common, "link_text": "", "pdf_url": "", "document_kind": ""}
            )

        if number % 25 == 0 or number == len(rows):
            save_checkpoint()
            print(f"Examined {number}/{len(rows)} detail pages", file=sys.stderr)

    if not rows and not args.output_csv.exists():
        save_checkpoint()

    pdf_rows = [row for row in output_rows if row["pdf_url"]]
    represented = {row["atlas_sample_id"] for row in output_rows if row["atlas_sample_id"]}
    print(f"Atlas detail pages represented: {len(represented)}")
    print(f"PDF links found: {len(pdf_rows)}")
    print(f"Unique PDF URLs: {len({row['pdf_url'] for row in pdf_rows})}")
    print(f"Discovery inventory: {args.output_csv}")
    return 0


DOWNLOAD_FIELDS = [
    "document_id",
    "atlas_sample_ids",
    "document_kind",
    "source_url",
    "local_pdf",
    "sha256",
    "size_bytes",
    "download_status",
    "downloaded_utc",
    "error",
]


def make_document_id(url: str, sample_ids: list[str], kind: str) -> str:
    url_name = unquote(Path(urlparse(url).path).name)
    fallback = sample_ids[0] if sample_ids else "document"
    stem = safe_name(Path(url_name).stem or fallback)
    url_tag = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
    return f"{stem}__{kind}__{url_tag}"


def command_download(args: argparse.Namespace) -> int:
    with args.inventory_csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("pdf_url", "").strip()]

    allowed = set(args.kinds.split(",")) if args.kinds != "all" else None
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if allowed is None or row.get("document_kind", "") in allowed:
            grouped[row["pdf_url"].strip()].append(row)

    args.pdf_dir.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, str]] = []
    for number, (url, linked_rows) in enumerate(sorted(grouped.items()), 1):
        sample_ids = sorted({row["atlas_sample_id"] for row in linked_rows})
        kinds = sorted({row.get("document_kind", "other_pdf") for row in linked_rows})
        kind = kinds[0] if len(kinds) == 1 else "mixed"
        document_id = make_document_id(url, sample_ids, kind)
        destination = args.pdf_dir / f"{document_id}.pdf"
        status = ""
        error = ""
        digest = ""
        size = ""
        try:
            if destination.is_file() and not args.overwrite:
                with destination.open("rb") as existing:
                    signature = existing.read(5)
                if signature != b"%PDF-":
                    raise ValueError("existing file does not begin with %PDF-")
                status = "existing"
            else:
                if number > 1 and args.delay:
                    time.sleep(args.delay)
                content, _ = fetch(url, args.timeout)
                if not content.startswith(b"%PDF-"):
                    raise ValueError("response does not begin with %PDF-")
                temporary = destination.with_suffix(".pdf.part")
                temporary.write_bytes(content)
                temporary.replace(destination)
                status = "downloaded"
            digest = sha256_file(destination)
            size = str(destination.stat().st_size)
        except Exception as exc:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"

        results.append(
            {
                "document_id": document_id,
                "atlas_sample_ids": ",".join(sample_ids),
                "document_kind": kind,
                "source_url": url,
                "local_pdf": str(destination.resolve()),
                "sha256": digest,
                "size_bytes": size,
                "download_status": status,
                "downloaded_utc": utc_now(),
                "error": error,
            }
        )
        print(f"{number}/{len(grouped)} {document_id}: {status}", file=sys.stderr)

    with args.output_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DOWNLOAD_FIELDS)
        writer.writeheader()
        writer.writerows(results)
    print(f"Unique documents considered: {len(results)}")
    print(f"Usable PDFs: {sum(row['download_status'] != 'error' for row in results)}")
    print(f"Download manifest: {args.output_manifest}")
    return 0


def command_tasks(args: argparse.Namespace) -> int:
    with args.download_manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    tasks: list[str] = []
    for row in rows:
        path = (
            args.pdf_dir / f"{row['document_id']}.pdf"
            if args.pdf_dir is not None
            else Path(row.get("local_pdf", ""))
        )
        if row.get("download_status") == "error" or not path.is_file():
            continue
        values = [
            row["document_id"],
            row.get("atlas_sample_ids", ""),
            str(path.resolve()),
            row.get("source_url", ""),
        ]
        if any("\t" in value or "\n" in value for value in values):
            raise ValueError(f"Task field contains a tab/newline: {values}")
        tasks.append("\t".join(values))
    args.output_tasks.parent.mkdir(parents=True, exist_ok=True)
    args.output_tasks.write_text("\n".join(tasks) + ("\n" if tasks else ""), encoding="utf-8")
    print(f"Tasks written: {len(tasks)}")
    print(f"Task file: {args.output_tasks}")
    return 0


def command_url_tasks(args: argparse.Namespace) -> int:
    """Write URL-based tasks; each worker downloads one temporary PDF."""
    with args.inventory_csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("pdf_url", "").strip()]
    allowed = set(args.kinds.split(",")) if args.kinds != "all" else None
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if allowed is None or row.get("document_kind", "") in allowed:
            grouped[row["pdf_url"].strip()].append(row)

    tasks: list[str] = []
    for url, linked_rows in sorted(grouped.items()):
        sample_ids = sorted({row["atlas_sample_id"] for row in linked_rows})
        kinds = sorted({row.get("document_kind", "other_pdf") for row in linked_rows})
        kind = kinds[0] if len(kinds) == 1 else "mixed"
        document_id = make_document_id(url, sample_ids, kind)
        values = [document_id, ",".join(sample_ids), url]
        if any("\t" in value or "\n" in value for value in values):
            raise ValueError(f"Task field contains a tab/newline: {values}")
        tasks.append("\t".join(values))

    args.output_tasks.parent.mkdir(parents=True, exist_ok=True)
    args.output_tasks.write_text("\n".join(tasks) + ("\n" if tasks else ""), encoding="utf-8")
    print(f"URL tasks written: {len(tasks)}")
    print(f"Task file: {args.output_tasks}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    discover = sub.add_parser("discover", help="Cache detail pages and inventory their PDF links")
    discover.add_argument("--atlas-csv", type=Path, required=True)
    discover.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional cache. Omit to process each detail page without retaining HTML.",
    )
    discover.add_argument("--output-csv", type=Path, required=True)
    discover.add_argument("--delay", type=float, default=0.5, help="Seconds between live requests")
    discover.add_argument("--timeout", type=float, default=60.0)
    discover.add_argument("--refresh", action="store_true", help="Redownload cached detail pages")
    discover.add_argument(
        "--restart",
        action="store_true",
        help="Ignore an existing output CSV and start the discovery inventory again",
    )
    discover.add_argument("--start-at", default="", help="Start at this Atlas sample ID")
    discover.add_argument("--limit", type=int, default=0, help="For testing; zero means all rows")
    discover.set_defaults(func=command_discover)

    download = sub.add_parser("download", help="Download unique PDFs in a discovery inventory")
    download.add_argument("--inventory-csv", type=Path, required=True)
    download.add_argument("--pdf-dir", type=Path, required=True)
    download.add_argument("--output-manifest", type=Path, required=True)
    download.add_argument(
        "--kinds",
        default="compendium",
        help="Comma-separated kinds (default: compendium), or all",
    )
    download.add_argument("--delay", type=float, default=0.5)
    download.add_argument("--timeout", type=float, default=120.0)
    download.add_argument("--overwrite", action="store_true")
    download.set_defaults(func=command_download)

    tasks = sub.add_parser("tasks", help="Build the Slurm TSV task file")
    tasks.add_argument("--download-manifest", type=Path, required=True)
    tasks.add_argument(
        "--pdf-dir",
        type=Path,
        default=None,
        help="Override machine-specific local_pdf paths using DOCUMENT_ID.pdf files here",
    )
    tasks.add_argument("--output-tasks", type=Path, required=True)
    tasks.set_defaults(func=command_tasks)

    url_tasks = sub.add_parser(
        "url-tasks",
        help="Build URL tasks whose workers download, parse, and delete one PDF",
    )
    url_tasks.add_argument("--inventory-csv", type=Path, required=True)
    url_tasks.add_argument("--output-tasks", type=Path, required=True)
    url_tasks.add_argument(
        "--kinds",
        default="compendium",
        help="Comma-separated kinds (default: compendium), or all",
    )
    url_tasks.set_defaults(func=command_url_tasks)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return args.func(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
