#!/usr/bin/env python3
"""Inventory and chop RELAB/Atlas evidence for lunar-soil I_s/FeO.

This is a candidate-discovery stage, not a final scientific adjudication.

Inputs
------
* RELAB spectrum/chemistry CSVs and their referenced PDS4 XML labels.
* Retained Lunar Sample Atlas ``json.gz`` page extractions.
* The two earlier incidental age/maturity CSVs, when present.

Outputs
-------
* ``existing_is_feo_records.csv``: values already recorded by earlier work.
* ``relab_is_feo_hits.csv``: numerical or qualitative RELAB metadata hits.
* ``atlas_is_feo_candidates.csv``: one row per Atlas page with I_s/FeO text.
* ``atlas_is_feo_review_chunks.jsonl``: compact inputs for local AI/manual review.
* ``is_feo_scan_manifest.json``: counts, parameters, and output hashes.

The program never transfers a parent value to a child/fraction.  It excludes
Bhatt-derived material by default.  Numerical values in candidate rows are raw
regex observations only; sample/value association is deliberately deferred.
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


PROGRAM_VERSION = "is-feo-candidate-collector-1.0"

# OCR/text extraction creates several spellings: Is, I_s, I s, I-sub-s, Iₛ.
IS_FEO_RE = re.compile(
    r"(?ix)"
    r"(?<![A-Za-z0-9])"
    r"(?:I\s*_?\s*s|I\s*[- ]?sub\s*[- ]?s|Iₛ)"
    r"\s*(?:/|\\|per)\s*Fe\s*O"
    r"(?![A-Za-z])"
)
MORRIS_INDEX_RE = re.compile(r"(?i)\bMorris(?:'s)?\s+(?:maturity\s+)?index\b")
MATURITY_RE = re.compile(
    r"(?i)\b(?:very\s+)?(?:im\s*mature|sub\s*mature|mature|maturity)\b"
)
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9.])([<>~≈]?\s*\d+(?:\.\d+)?)(?![A-Za-z0-9.])")
SAMPLE_ID_RE = re.compile(r"(?<!\d)([1-7]\d{4}(?:,\d{1,3})?)(?!\d)")
PAGE_HEADER_RE = re.compile(r"^=== PDF PAGE\s+(\d+);", re.M)
EXCLUDED_DEFAULT = r"(?i)bhatt"

EXISTING_COLUMNS = [
    "sample_id",
    "value_text",
    "qualifier",
    "units_or_scale",
    "scope",
    "source_url",
    "source_locator",
    "evidence",
    "needs_review",
    "source_table",
]

RELAB_COLUMNS = [
    "hit_id",
    "match_kind",
    "raw_numeric_tokens",
    "measurement_ids",
    "specimen_ids",
    "source_file",
    "source_file_sha256",
    "line_number",
    "evidence",
    "notes",
]

ATLAS_COLUMNS = [
    "chunk_id",
    "chunk_text_sha256",
    "candidate_kind",
    "document_id",
    "atlas_document_sample_ids",
    "pdf_page",
    "source_url",
    "source_pdf_sha256",
    "raw_numeric_tokens",
    "sample_ids_seen_in_chunk",
    "evidence",
    "needs_review",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan RELAB metadata and parsed Atlas PDFs for I_s/FeO evidence."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="M3_RELAB_Atlas_Crosswalk_v1 directory.",
    )
    parser.add_argument(
        "--relab-index",
        type=Path,
        default=None,
        help="RELAB spectrum index CSV; defaults under project root.",
    )
    parser.add_argument(
        "--relab-chemistry",
        type=Path,
        default=None,
        help="RELAB chemistry CSV; defaults under project root.",
    )
    parser.add_argument(
        "--relab-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Root containing original RELAB XML metadata. Repeatable. By default "
            "the script also tries the sibling cartApollo2026 directory."
        ),
    )
    parser.add_argument(
        "--atlas-json-dir",
        type=Path,
        default=None,
        help="Atlas parsed json_gz directory; defaults to data/parsed/json_gz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to data/is_feo_v1.",
    )
    parser.add_argument(
        "--context-lines",
        type=int,
        default=5,
        help="Lines retained before and after an Atlas hit (default: 5).",
    )
    parser.add_argument(
        "--exclude-source-regex",
        default=EXCLUDED_DEFAULT,
        help="Case-insensitive source/path exclusion regex (default excludes Bhatt).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory manifest/files.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_space(value: str) -> str:
    return " ".join(value.replace("\x00", " ").replace("\xa0", " ").split())


def unique_join(values: Iterable[str]) -> str:
    return ";".join(dict.fromkeys(v.strip() for v in values if v and v.strip()))


def atomic_write_csv(path: Path, rows: Iterable[dict[str, Any]], columns: list[str]) -> None:
    temporary = path.with_name(path.name + ".part")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    temporary.replace(path)


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def resolve_default(root: Path, explicit: Path | None, relative: str) -> Path:
    return explicit.expanduser().resolve() if explicit else root / relative


def display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def is_is_feo_property(row: dict[str, str]) -> bool:
    joined = " ".join(
        str(row.get(key, ""))
        for key in ("property", "unit", "units_or_scale", "evidence", "note")
    )
    return bool(IS_FEO_RE.search(joined)) or (
        "maturity" in joined.lower() and "feo" in joined.lower()
    )


def split_qualifier(value: str) -> tuple[str, str]:
    text = clean_space(value)
    match = re.match(r"^([<>~≈])\s*(.*)$", text)
    return (match.group(2), match.group(1)) if match else (text, "")


def load_existing_records(project_root: Path, excluded: re.Pattern[str]) -> list[dict[str, str]]:
    paths = [
        project_root / "data/ai_resolution_v1/incidental_al2o3_age_maturity.csv",
        project_root / "data/ai_resolution_v1/missing_case_incidental_al2o3_age_maturity.csv",
    ]
    results: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    for path in paths:
        for row in read_csv(path):
            if not is_is_feo_property(row):
                continue
            source_url = row.get("source_url", "")
            evidence = row.get("evidence", "") or row.get("note", "")
            if excluded.search(" ".join([str(path), source_url, evidence])):
                continue
            value, qualifier = split_qualifier(
                row.get("value_text", "") or row.get("values", "")
            )
            result = {
                "sample_id": row.get("sample_id", "") or row.get("apollo_parent_id", ""),
                "value_text": value,
                "qualifier": qualifier,
                "units_or_scale": row.get("unit", "") or row.get("units_or_scale", ""),
                "scope": row.get("scope", ""),
                "source_url": source_url,
                "source_locator": row.get("pdf_page", "") or row.get("source_locator", ""),
                "evidence": evidence,
                "needs_review": row.get("needs_review", "") or "not_stated",
                "source_table": str(path.relative_to(project_root)),
            }
            key = tuple(result[column] for column in EXISTING_COLUMNS[:-1])
            if key not in seen:
                seen.add(key)
                results.append(result)
    return sorted(results, key=lambda row: (row["sample_id"], row["value_text"]))


def numeric_tokens_near_is_feo(text: str) -> list[str]:
    values: list[str] = []
    for match in IS_FEO_RE.finditer(text):
        neighborhood = text[max(0, match.start() - 220) : match.end() + 300]
        for number in NUMBER_RE.findall(neighborhood):
            token = clean_space(number)
            core = re.sub(r"[^0-9.]", "", token)
            if not core:
                continue
            try:
                numeric = float(core)
            except ValueError:
                continue
            integer_part = core.split(".", 1)[0]
            # Five-digit Apollo IDs, publication years, and bibliography page
            # numbers are not plausible maturity-index values.  The broad
            # 0--200 interval is a candidate filter, not scientific validation.
            if len(integer_part) == 5 or 1900 <= numeric <= 2099 or numeric > 200:
                continue
            if token not in values:
                values.append(token)
    return values


def classify_hit(text: str) -> str:
    if IS_FEO_RE.search(text):
        return "explicit_is_feo"
    if MORRIS_INDEX_RE.search(text):
        return "morris_index"
    if MATURITY_RE.search(text):
        return "qualitative_maturity"
    return ""


def csv_row_text(row: dict[str, str]) -> str:
    omitted = {
        "measurement_data_sha256",
        "source_metadata_sha256",
        "chemistry_data_sha256",
    }
    return " | ".join(f"{key}={value}" for key, value in row.items() if value and key not in omitted)


def scan_relab_csv_rows(
    index_path: Path, chemistry_path: Path, excluded: re.Pattern[str]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, set[str]]], set[Path]]:
    hits: list[dict[str, Any]] = []
    metadata_links: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"measurement_ids": set(), "specimen_ids": set()}
    )
    referenced_paths: set[Path] = set()

    for table_kind, path in (("spectrum_index", index_path), ("chemistry", chemistry_path)):
        for row_number, row in enumerate(read_csv(path), 2):
            measurement_id = row.get("measurement_id", "") or row.get("chemistry_product_id", "")
            specimen_id = row.get("specimen_id", "")
            metadata_value = row.get("source_metadata_file", "").strip()
            if metadata_value:
                source_path = Path(metadata_value).expanduser()
                referenced_paths.add(source_path)
                link = metadata_links[str(source_path)]
                link["measurement_ids"].add(measurement_id)
                link["specimen_ids"].add(specimen_id)

            text = csv_row_text(row)
            kind = classify_hit(text)
            if not kind or excluded.search(text):
                continue
            hits.append(
                {
                    "match_kind": kind,
                    "raw_numeric_tokens": unique_join(numeric_tokens_near_is_feo(text)),
                    "measurement_ids": measurement_id,
                    "specimen_ids": specimen_id,
                    "source_file": str(path),
                    "source_file_sha256": sha256_file(path),
                    "line_number": row_number,
                    "evidence": clean_space(text),
                    "notes": f"Hit in extracted RELAB {table_kind} row",
                }
            )
    return hits, metadata_links, referenced_paths


def metadata_candidates(roots: list[Path], referenced: set[Path]) -> list[Path]:
    candidates: dict[str, Path] = {}
    basenames: dict[str, list[Path]] = defaultdict(list)
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.xml"):
            if path.is_file():
                resolved = path.resolve()
                candidates[str(resolved)] = resolved
                basenames[path.name].append(resolved)
    for path in referenced:
        if path.is_file():
            candidates[str(path.resolve())] = path.resolve()
        elif len(basenames.get(path.name, [])) == 1:
            found = basenames[path.name][0]
            candidates[str(found)] = found
    return sorted(candidates.values(), key=str)


def context_for_line(lines: list[str], index: int, radius: int = 2) -> str:
    start = max(0, index - radius)
    end = min(len(lines), index + radius + 1)
    return clean_space(" ".join(lines[start:end]))


def scan_relab_xml(
    paths: list[Path],
    links: dict[str, dict[str, set[str]]],
    excluded: re.Pattern[str],
) -> tuple[list[dict[str, Any]], int]:
    hits: list[dict[str, Any]] = []
    failed = 0
    basename_links: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"measurement_ids": set(), "specimen_ids": set()}
    )
    for source_name, link in links.items():
        name = Path(source_name).name
        basename_links[name]["measurement_ids"].update(link["measurement_ids"])
        basename_links[name]["specimen_ids"].update(link["specimen_ids"])

    for path in paths:
        if excluded.search(str(path)):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            failed += 1
            continue
        lines = text.splitlines()
        digest = sha256_file(path)
        link = links.get(str(path), links.get(str(path.resolve()), basename_links[path.name]))
        seen_contexts: set[tuple[str, str]] = set()
        for line_index, line in enumerate(lines):
            context = context_for_line(lines, line_index)
            kind = classify_hit(context)
            if not kind or excluded.search(context):
                continue
            key = (kind, context.lower())
            if key in seen_contexts:
                continue
            seen_contexts.add(key)
            hits.append(
                {
                    "match_kind": kind,
                    "raw_numeric_tokens": unique_join(numeric_tokens_near_is_feo(context)),
                    "measurement_ids": unique_join(sorted(link["measurement_ids"])),
                    "specimen_ids": unique_join(sorted(link["specimen_ids"])),
                    "source_file": str(path),
                    "source_file_sha256": digest,
                    "line_number": line_index + 1,
                    "evidence": context,
                    "notes": "Hit in original RELAB PDS4 XML label",
                }
            )
    return hits, failed


def assign_hit_ids(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for row in rows:
        key = (
            row.get("match_kind", ""),
            row.get("measurement_ids", ""),
            row.get("specimen_ids", ""),
            row.get("source_file_sha256", ""),
            str(row.get("line_number", "")),
            row.get("evidence", "").lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        item = dict(row)
        item["hit_id"] = f"relab-hit-{len(unique) + 1:05d}"
        unique.append(item)
    return unique


def merge_ranges(indices: list[int], line_count: int, radius: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for index in indices:
        start = max(0, index - radius)
        end = min(line_count, index + radius + 1)
        if ranges and start <= ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
        else:
            ranges.append((start, end))
    return ranges


def atlas_page_chunk(text: str, radius: int) -> str:
    lines = [line.rstrip() for line in text.replace("\x00", "").splitlines()]
    hit_indices = [index for index, line in enumerate(lines) if IS_FEO_RE.search(line)]
    if not hit_indices:
        return ""
    pieces: list[str] = []
    for start, end in merge_ranges(hit_indices, len(lines), radius):
        piece = "\n".join(lines[start:end]).strip()
        if piece and piece not in pieces:
            pieces.append(piece)
    return "\n\n[... separate passage on same PDF page ...]\n\n".join(pieces)


def atlas_candidate_kind(evidence: str, raw_values: str) -> str:
    """Separate likely measurements from citations and graph-axis labels."""
    lower = clean_space(evidence).lower()
    occurrence_count = len(IS_FEO_RE.findall(evidence))
    if (
        occurrence_count == 1
        and "surface exposure" in lower
        and "compilation" in lower
        and "morris" in lower
    ):
        return "citation_only"
    if not raw_values:
        return "text_only_or_citation"
    if "respectively" in lower:
        return "numeric_statement"
    for match in IS_FEO_RE.finditer(evidence):
        window = clean_space(
            evidence[max(0, match.start() - 180) : match.end() + 220]
        )
        before = window[: max(0, window.lower().find(clean_space(match.group(0)).lower()))]
        after = window[len(before) + len(clean_space(match.group(0))) :]
        if re.search(r"^\s*[,):]?\s*=\s*[<>~≈]?\s*\d", after):
            return "numeric_statement"
        if re.search(r"(?i)\b(?:is|are|was|were|as)\s*[<>~≈]?\s*\d", window):
            return "numeric_statement"
        if re.search(r"(?i)[<>~≈]?\s*\d+(?:\.\d+)?\s*$", before):
            return "numeric_statement"
    if re.search(r"(?i)\btable\b", evidence):
        return "numeric_table_or_layout"
    return "numeric_graph_or_ambiguous"


def scan_atlas_json(
    json_dir: Path, radius: int, excluded: re.Pattern[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    failed = 0
    for path in sorted(json_dir.glob("*.json.gz")):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                document = json.load(handle)
        except (OSError, json.JSONDecodeError, EOFError):
            failed += 1
            continue
        source = document.get("source", {})
        document_id = str(source.get("document_id", path.name.removesuffix(".json.gz")))
        source_url = str(source.get("source_url", ""))
        source_hash = str(source.get("source_pdf_sha256", ""))
        document_samples = [str(value) for value in source.get("atlas_sample_ids", [])]
        if excluded.search(" ".join([str(path), document_id, source_url])):
            continue
        for page in document.get("pages", []):
            page_text = str(page.get("text", ""))
            if not IS_FEO_RE.search(page_text):
                continue
            evidence = atlas_page_chunk(page_text, radius)
            if not evidence or excluded.search(evidence):
                continue
            page_number = int(page.get("page_number", 0) or 0)
            chunk_id = f"{document_id}__page_{page_number:03d}"
            chunk_text_sha256 = hashlib.sha256(evidence.encode("utf-8")).hexdigest()
            samples_seen = unique_join(SAMPLE_ID_RE.findall(evidence))
            raw_values = unique_join(numeric_tokens_near_is_feo(evidence))
            candidate_kind = atlas_candidate_kind(evidence, raw_values)
            row = {
                "chunk_id": chunk_id,
                "chunk_text_sha256": chunk_text_sha256,
                "candidate_kind": candidate_kind,
                "document_id": document_id,
                "atlas_document_sample_ids": unique_join(document_samples),
                "pdf_page": page_number,
                "source_url": source_url,
                "source_pdf_sha256": source_hash,
                "raw_numeric_tokens": raw_values,
                "sample_ids_seen_in_chunk": samples_seen,
                "evidence": evidence,
                "needs_review": "yes",
                "notes": (
                    "Candidate page only; do not infer a value from a graph and do not "
                    "transfer values between parent, child, split, or grain-size fraction."
                ),
            }
            rows.append(row)
            if candidate_kind in {"numeric_statement", "numeric_table_or_layout"}:
                chunks.append(
                    {
                        "schema": "atlas-is-feo-review-chunk-1.0",
                        "chunk_id": chunk_id,
                        "chunk_text_sha256": chunk_text_sha256,
                        "document_id": document_id,
                        "atlas_document_sample_ids": document_samples,
                        "pdf_page": page_number,
                        "source_url": source_url,
                        "source_pdf_sha256": source_hash,
                        "raw_numeric_tokens": raw_values.split(";"),
                        "text": evidence,
                    }
                )
    return rows, chunks, failed


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    if not project_root.is_dir():
        print(f"ERROR: project root not found: {project_root}", file=sys.stderr)
        return 2

    index_path = resolve_default(
        project_root, args.relab_index, "relab_raw_native_2026-10-04/relab_spectrum_index.csv"
    )
    chemistry_path = resolve_default(
        project_root, args.relab_chemistry, "relab_raw_native_2026-10-04/relab_chemistry.csv"
    )
    atlas_json_dir = resolve_default(
        project_root, args.atlas_json_dir, "data/parsed/json_gz"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else project_root / "data/is_feo_v1"
    )
    manifest_path = output_dir / "is_feo_scan_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        print(
            f"ERROR: {manifest_path} exists; use --overwrite to replace this version's outputs.",
            file=sys.stderr,
        )
        return 2
    for required in (index_path, chemistry_path):
        if not required.is_file():
            print(f"ERROR: required RELAB table not found: {required}", file=sys.stderr)
            return 2
    if not atlas_json_dir.is_dir():
        print(f"ERROR: Atlas parsed directory not found: {atlas_json_dir}", file=sys.stderr)
        return 2
    if args.context_lines < 1:
        print("ERROR: --context-lines must be at least 1", file=sys.stderr)
        return 2

    try:
        excluded = re.compile(args.exclude_source_regex, re.I)
    except re.error as exc:
        print(f"ERROR: invalid --exclude-source-regex: {exc}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    relab_roots = [path.expanduser().resolve() for path in args.relab_root]
    default_relab_root = project_root.parent / "cartApollo2026"
    if default_relab_root.is_dir() and default_relab_root.resolve() not in relab_roots:
        relab_roots.append(default_relab_root.resolve())

    existing = load_existing_records(project_root, excluded)
    csv_hits, links, referenced_paths = scan_relab_csv_rows(
        index_path, chemistry_path, excluded
    )
    xml_paths = metadata_candidates(relab_roots, referenced_paths)
    xml_hits, relab_failures = scan_relab_xml(xml_paths, links, excluded)
    relab_hits = assign_hit_ids(csv_hits + xml_hits)
    atlas_rows, atlas_chunks, atlas_failures = scan_atlas_json(
        atlas_json_dir, args.context_lines, excluded
    )

    output_paths = {
        "existing": output_dir / "existing_is_feo_records.csv",
        "relab": output_dir / "relab_is_feo_hits.csv",
        "atlas": output_dir / "atlas_is_feo_candidates.csv",
        "chunks": output_dir / "atlas_is_feo_review_chunks.jsonl",
    }
    atomic_write_csv(output_paths["existing"], existing, EXISTING_COLUMNS)
    atomic_write_csv(output_paths["relab"], relab_hits, RELAB_COLUMNS)
    atomic_write_csv(output_paths["atlas"], atlas_rows, ATLAS_COLUMNS)
    atomic_write_text(
        output_paths["chunks"],
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in atlas_chunks),
    )

    manifest = {
        "schema": PROGRAM_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "inputs": {
            "relab_index": str(index_path),
            "relab_chemistry": str(chemistry_path),
            "relab_roots": [str(path) for path in relab_roots],
            "atlas_json_dir": str(atlas_json_dir),
        },
        "rules": {
            "exclude_source_regex": args.exclude_source_regex,
            "parent_child_transfer": "forbidden",
            "graph_reading": "forbidden",
            "candidate_values_are_confirmed": False,
        },
        "counts": {
            "previously_recorded_rows": len(existing),
            "relab_xml_files_scanned": len(xml_paths),
            "relab_hits": len(relab_hits),
            "relab_read_failures": relab_failures,
            "atlas_json_files_scanned": len(list(atlas_json_dir.glob("*.json.gz"))),
            "atlas_candidate_pages": len(atlas_rows),
            "atlas_numeric_review_chunks": len(atlas_chunks),
            "atlas_read_failures": atlas_failures,
        },
        "outputs": {},
    }
    for name, path in output_paths.items():
        manifest["outputs"][name] = {
            "path": display_path(path, project_root),
            "sha256": sha256_file(path),
        }
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2) + "\n")

    print(f"Previously recorded I_s/FeO rows: {len(existing)}")
    print(f"RELAB XML metadata files scanned: {len(xml_paths)}")
    print(f"RELAB metadata hits: {len(relab_hits)}")
    print(f"Atlas parsed PDF files scanned: {manifest['counts']['atlas_json_files_scanned']}")
    print(f"Atlas candidate pages: {len(atlas_rows)}")
    print(f"Atlas numeric review chunks: {len(atlas_chunks)}")
    print(f"Outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
