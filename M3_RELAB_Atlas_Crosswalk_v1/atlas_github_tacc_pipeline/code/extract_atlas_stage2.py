#!/usr/bin/env python3
"""Build traceable Stage-2 Atlas tables from Stage-1 JSON.GZ files.

This program never guesses missing chemistry.  It uses PDF word coordinates to
recover numeric FeO cells, links each cell to the most plausible printed column
heading, and marks ambiguous cases for review.  Narrative pages are exported
separately for later human or language-model interpretation.
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


VERSION = "atlas-stage2-0.1.0"
NUMBER_RE = re.compile(r"^[<>~≈]?[-+]?(?:\d+(?:\.\d*)?|\.\d+)$")
AUTHOR_YEAR_RE = re.compile(r"(?i)^(?:[A-Z][A-Za-z’'\-]*|[A-Z]{2,})\d{2,4}[a-z]?$|^LSPET\d{2}$")
METHOD_RE = re.compile(
    r"\(\s*([a-z])\s*\)\s*(.*?)(?=,\s*\(\s*[a-z]\s*\)|$)", re.I
)
MINERAL_PHASE_RE = re.compile(
    r"(?i)\b(?:mineral|pyroxene|olivine|plagioclase|ilmenite|spinel|glass|"
    r"clinopyroxene|orthopyroxene|feldspar)\b"
)
SAMPLE_COMPOSITION_RE = re.compile(
    r"(?i)\b(?:chemical|major[- ]element|bulk)\s+composition\s+of\s+(?:sample\s+)?\d{4,6}\b"
)
REFERENCE_HEADING_RE = re.compile(r"(?i)^\s*(?:references|bibliography)\s*$")


DOCUMENT_COLUMNS = [
    "document_id", "atlas_sample_ids", "source_url", "source_pdf_sha256",
    "source_pdf_size_bytes", "page_count", "pages_with_extractable_text",
    "relevant_pages", "evidence_block_count", "needs_ocr", "parser_name",
]

FEO_COLUMNS = [
    "measurement_id", "document_id", "atlas_sample_id", "pdf_page",
    "value_wt_percent", "value_text", "qualifier", "uncertainty_wt_percent",
    "reference", "column_detail", "analytical_method_code",
    "analytical_method", "sample_scope", "table_heading", "evidence_text",
    "source_url", "source_pdf_sha256", "x0", "y0", "confidence",
    "needs_review", "review_reason",
]

REVIEW_COLUMNS = [
    "document_id", "atlas_sample_ids", "pdf_page", "issue", "details",
    "source_url", "source_pdf_sha256",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-dir", type=Path, required=True,
                        help="Directory containing json_gz/ from Stage 1")
    parser.add_argument("--atlas-csv", type=Path,
                        help="Optional atlas_samples.csv from the Atlas HTML index")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_gz(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: Iterable[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in columns})
    temporary.replace(path)


def words_at_y(words: list[dict[str, Any]], y: float, tolerance: float = 1.5) -> list[dict[str, Any]]:
    return sorted(
        (word for word in words if abs(float(word["y0"]) - y) <= tolerance),
        key=lambda word: (float(word["x0"]), int(word.get("word", 0))),
    )


def method_map(technique_rows: list[dict[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in technique_rows:
        text = re.sub(r"\(\s*([a-z])\s+\)", r"(\1)", str(row.get("text", "")), flags=re.I)
        text = re.sub(r"(?i)^.*?\b(?:technique|method)\s*:\s*", "", text)
        for code, value in METHOD_RE.findall(text):
            result[code.lower()] = value.strip(" ,.;")
    return result


def clean_numeric_token(text: str) -> tuple[str, str] | None:
    value = text.strip().rstrip("*,;:")
    if not NUMBER_RE.fullmatch(value):
        return None
    qualifier = ""
    if value and value[0] in "<>~≈":
        qualifier, value = value[0], value[1:]
    try:
        float(value)
    except ValueError:
        return None
    return value, qualifier


def group_method_code(tokens: list[dict[str, Any]]) -> str:
    joined = " ".join(str(token["text"]) for token in tokens)
    match = re.search(r"\(\s*([a-z])\s*\)", joined, re.I)
    return match.group(1).lower() if match else ""


def table_heading(words: list[dict[str, Any]], reference_y: float) -> str:
    candidates = [word for word in words if float(word["y0"]) < reference_y - 2]
    if not candidates:
        return ""
    rows: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for word in candidates:
        rows[(int(word.get("block", 0)), int(word.get("line", 0)))].append(word)
    built: list[tuple[float, str]] = []
    for members in rows.values():
        members.sort(key=lambda word: float(word["x0"]))
        y = min(float(word["y0"]) for word in members)
        built.append((y, " ".join(str(word["text"]) for word in members)))
    built.sort()
    nearby = [text for y, text in built if reference_y - 80 <= y < reference_y - 2]
    return " | ".join(nearby[-2:])


def classify_scope(heading: str) -> str:
    if MINERAL_PHASE_RE.search(heading):
        return "mineral_phase"
    if SAMPLE_COMPOSITION_RE.search(heading):
        return "bulk_or_whole_sample"
    return "unspecified"


def reference_headers(words: list[dict[str, Any]], reference_y: float, feo_y: float) -> list[dict[str, Any]]:
    """Recover printed reference headings on the visual ``reference`` row.

    Simple author-year tokens are reliable.  A wrapped construction such as
    ``Haramura in Compston70`` is retained as one complex heading and later
    sent to review instead of being silently assigned high confidence.
    """
    row = words_at_y(words, reference_y)
    row = [word for word in row if float(word["x0"]) > 85 and float(word["y0"]) < feo_y - 2]
    headers: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for word in row:
        text = str(word["text"]).strip(" ,;:")
        if not text or text.lower() == "reference":
            continue
        is_year = bool(AUTHOR_YEAR_RE.fullmatch(text))
        is_author = bool(re.fullmatch(r"[A-Z][A-Za-z’'\-]{3,}", text))
        connector = text.lower() in {"in", "and", "&", "et", "al", "al."}
        if is_year and current and current["parts"][-1].lower() in {"in", "and", "&", "et", "al", "al."}:
            current["parts"].append(text)
            current["x1"] = float(word["x1"])
            current["complex"] = True
        elif is_year or is_author:
            if current:
                headers.append(current)
            current = {
                "parts": [text], "x0": float(word["x0"]), "x1": float(word["x1"]),
                "complex": not is_year,
            }
        elif connector and current:
            current["parts"].append(text)
            current["x1"] = float(word["x1"])
            current["complex"] = True
        elif current:
            current["parts"].append(text)
            current["x1"] = float(word["x1"])
            current["complex"] = True
    if current:
        headers.append(current)
    for header in headers:
        header["text"] = " ".join(header.pop("parts"))
    return headers


def map_reference(headers: list[dict[str, Any]], x0: float) -> dict[str, Any] | None:
    if not headers:
        return None
    preceding = [header for header in headers if header["x0"] <= x0 + 3]
    if preceding:
        return preceding[-1]
    return min(headers, key=lambda header: abs(header["x0"] - x0))


def column_detail(words: list[dict[str, Any]], reference_y: float, feo_y: float, x0: float) -> str:
    ignored = {"reference", "weight", "mg", "g", "%"}
    values: list[str] = []
    for word in words:
        y = float(word["y0"])
        text = str(word["text"]).strip(" ,;:")
        if not (reference_y + 2 < y < feo_y - 2):
            continue
        if abs(float(word["x0"]) - x0) > 14:
            continue
        if not text or text.lower() in ignored or AUTHOR_YEAR_RE.fullmatch(text):
            continue
        if NUMBER_RE.fullmatch(text):
            continue
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{1,20}", text):
            values.append(text)
    return " ".join(dict.fromkeys(values))


def extract_feo_from_page(
    document: dict[str, Any], page: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = document.get("source", {})
    layout = page.get("chemistry_layout", {})
    words = page.get("word_coordinates", [])
    sample_ids = list(source.get("atlas_sample_ids", []))
    records: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    methods = method_map(layout.get("technique_rows", []))

    for feo_row in layout.get("feo_rows", []):
        y = float(feo_row["y0"])
        prior_refs = [row for row in layout.get("reference_rows", []) if float(row["y0"]) < y]
        ref_row = max(prior_refs, key=lambda row: float(row["y0"]), default=None)
        row_words = words_at_y(words, y)
        numeric: list[tuple[int, dict[str, Any], str, str]] = []
        for index, word in enumerate(row_words):
            parsed = clean_numeric_token(str(word["text"]))
            if parsed:
                numeric.append((index, word, parsed[0], parsed[1]))

        # A number immediately following ± is uncertainty, not another measurement.
        measurement_items: list[tuple[int, dict[str, Any], str, str, str]] = []
        for position, (index, word, value, qualifier) in enumerate(numeric):
            previous_text = str(row_words[index - 1]["text"]) if index else ""
            if previous_text in {"±", "+/-"}:
                continue
            next_index = numeric[position + 1][0] if position + 1 < len(numeric) else len(row_words)
            between = row_words[index + 1:next_index]
            uncertainty = ""
            for offset, token in enumerate(between[:-1]):
                if str(token["text"]) in {"±", "+/-"}:
                    candidate = clean_numeric_token(str(between[offset + 1]["text"]))
                    if candidate:
                        uncertainty = candidate[0]
            measurement_items.append((index, word, value, qualifier, uncertainty))

        if not measurement_items:
            # Blank FeO rows are useful negative evidence but are not measurements.
            continue

        reference_y = float(ref_row["y0"]) if ref_row else y
        headers = reference_headers(words, reference_y, y) if ref_row else []
        heading = table_heading(words, reference_y) if ref_row else ""
        scope = classify_scope(heading)
        evidence = " ".join(str(word["text"]) for word in row_words)

        for position, (index, word, value, qualifier, uncertainty) in enumerate(measurement_items):
            next_index = measurement_items[position + 1][0] if position + 1 < len(measurement_items) else len(row_words)
            code = group_method_code(row_words[index + 1:next_index])
            reference_header = map_reference(headers, float(word["x0"]))
            reference = str(reference_header["text"]) if reference_header else ""
            reasons: list[str] = []
            if not ref_row:
                reasons.append("no_reference_header_above_feo_row")
            if not reference:
                reasons.append("value_not_mapped_to_reference")
            if reference_header and reference_header.get("complex"):
                reasons.append("complex_reference_header")
            if len(sample_ids) != 1:
                reasons.append("document_has_zero_or_multiple_sample_ids")
            if scope == "unspecified":
                reasons.append("sample_scope_unspecified")
            if code and code not in methods:
                reasons.append("method_code_not_resolved")
            confidence = "high" if not reasons else ("medium" if reference and ref_row else "low")
            records.append({
                "document_id": source.get("document_id", ""),
                "atlas_sample_id": sample_ids[0] if len(sample_ids) == 1 else ",".join(sample_ids),
                "pdf_page": page.get("page_number", ""),
                "value_wt_percent": value,
                "value_text": (qualifier + value),
                "qualifier": qualifier,
                "uncertainty_wt_percent": uncertainty,
                "reference": reference,
                "column_detail": column_detail(words, reference_y, y, float(word["x0"])) if ref_row else "",
                "analytical_method_code": code,
                "analytical_method": methods.get(code, ""),
                "sample_scope": scope,
                "table_heading": heading,
                "evidence_text": evidence,
                "source_url": source.get("source_url", ""),
                "source_pdf_sha256": source.get("source_pdf_sha256", ""),
                "x0": word.get("x0", ""),
                "y0": word.get("y0", ""),
                "confidence": confidence,
                "needs_review": "yes" if reasons else "no",
                "review_reason": ";".join(reasons),
            })

        if not ref_row:
            reviews.append({
                "document_id": source.get("document_id", ""),
                "atlas_sample_ids": ",".join(sample_ids),
                "pdf_page": page.get("page_number", ""),
                "issue": "feo_without_reference_header",
                "details": evidence,
                "source_url": source.get("source_url", ""),
                "source_pdf_sha256": source.get("source_pdf_sha256", ""),
            })
    return records, reviews


def reference_start_page(document: dict[str, Any]) -> int | None:
    for page in document.get("pages", []):
        if any(REFERENCE_HEADING_RE.fullmatch(line) for line in str(page.get("text", "")).splitlines()):
            return int(page["page_number"])
    return None


def narrative_pages(document: dict[str, Any]) -> list[dict[str, Any]]:
    source = document.get("source", {})
    start = reference_start_page(document)
    output = []
    wanted = {"material", "mineralogy", "age", "treatment"}
    for page in document.get("pages", []):
        categories = sorted(wanted.intersection(page.get("categories", [])))
        if not categories:
            continue
        page_number = int(page["page_number"])
        if start is not None and page_number >= start:
            continue
        text = str(page.get("text", "")).strip()
        if not text:
            continue
        output.append({
            "document_id": source.get("document_id", ""),
            "atlas_sample_ids": source.get("atlas_sample_ids", []),
            "pdf_page": page_number,
            "categories": categories,
            "text": text,
            "text_sha256": page.get("text_sha256", ""),
            "source_url": source.get("source_url", ""),
            "source_pdf_sha256": source.get("source_pdf_sha256", ""),
        })
    return output


def document_row(document: dict[str, Any]) -> dict[str, Any]:
    source = document.get("source", {})
    summary = document.get("summary", {})
    parser = document.get("parser", {})
    return {
        "document_id": source.get("document_id", ""),
        "atlas_sample_ids": ",".join(source.get("atlas_sample_ids", [])),
        "source_url": source.get("source_url", ""),
        "source_pdf_sha256": source.get("source_pdf_sha256", ""),
        "source_pdf_size_bytes": source.get("source_pdf_size_bytes", ""),
        "page_count": summary.get("page_count", ""),
        "pages_with_extractable_text": summary.get("pages_with_extractable_text", ""),
        "relevant_pages": summary.get("relevant_pages", ""),
        "evidence_block_count": summary.get("evidence_block_count", ""),
        "needs_ocr": summary.get("needs_ocr", ""),
        "parser_name": parser.get("name", ""),
    }


def enrich_atlas_samples(path: Path, feo_rows: list[dict[str, Any]], document_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        source_rows = list(reader)
        source_columns = list(reader.fieldnames or [])
    feo_by_sample: dict[str, list[float]] = defaultdict(list)
    for row in feo_rows:
        if row["sample_scope"] != "bulk_or_whole_sample":
            continue
        try:
            feo_by_sample[str(row["atlas_sample_id"])].append(float(row["value_wt_percent"]))
        except (TypeError, ValueError):
            pass
    docs_by_sample: dict[str, int] = defaultdict(int)
    for row in document_rows:
        for sample_id in str(row["atlas_sample_ids"]).split(","):
            if sample_id:
                docs_by_sample[sample_id] += 1
    added = ["parsed_document_count", "bulk_feo_measurement_count", "bulk_feo_min_wt_percent", "bulk_feo_max_wt_percent"]
    for row in source_rows:
        sample_id = str(row.get("atlas_sample_id", ""))
        values = feo_by_sample.get(sample_id, [])
        row.update({
            "parsed_document_count": docs_by_sample.get(sample_id, 0),
            "bulk_feo_measurement_count": len(values),
            "bulk_feo_min_wt_percent": min(values) if values else "",
            "bulk_feo_max_wt_percent": max(values) if values else "",
        })
    return source_rows, source_columns + added


def main() -> int:
    args = parse_args()
    json_dir = args.parsed_dir / "json_gz"
    files = sorted(json_dir.glob("*.json.gz"))
    if not files:
        print(f"ERROR: no *.json.gz files in {json_dir}", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "stage2_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        print(f"ERROR: output already exists; use --overwrite: {manifest_path}", file=sys.stderr)
        return 2

    documents: list[dict[str, Any]] = []
    feo: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    narratives: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for path in files:
        try:
            document = load_json_gz(path)
            documents.append(document_row(document))
            narratives.extend(narrative_pages(document))
            for page in document.get("pages", []):
                found, page_reviews = extract_feo_from_page(document, page)
                feo.extend(found)
                reviews.extend(page_reviews)
        except Exception as exc:  # keep a bounded, auditable failure list
            errors.append({"file": path.name, "error": f"{type(exc).__name__}: {exc}"})

    feo.sort(key=lambda row: (str(row["atlas_sample_id"]), str(row["document_id"]), int(row["pdf_page"]), float(row["x0"])))
    for index, row in enumerate(feo, 1):
        row["measurement_id"] = index
        if row["needs_review"] == "yes":
            reviews.append({
                "document_id": row["document_id"],
                "atlas_sample_ids": row["atlas_sample_id"],
                "pdf_page": row["pdf_page"],
                "issue": "feo_measurement_needs_review",
                "details": f"measurement_id={index}; {row['review_reason']}; {row['evidence_text']}",
                "source_url": row["source_url"],
                "source_pdf_sha256": row["source_pdf_sha256"],
            })

    write_csv(args.output_dir / "atlas_documents.csv", documents, DOCUMENT_COLUMNS)
    write_csv(args.output_dir / "atlas_feo_measurements.csv", feo, FEO_COLUMNS)
    write_csv(args.output_dir / "atlas_review_queue.csv", reviews, REVIEW_COLUMNS)
    atomic_text(
        args.output_dir / "atlas_narrative_pages.jsonl",
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in narratives),
    )
    if args.atlas_csv:
        enriched, columns = enrich_atlas_samples(args.atlas_csv, feo, documents)
        write_csv(args.output_dir / "atlas_samples_enriched.csv", enriched, columns)

    outputs = {}
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file() and path.name != manifest_path.name:
            outputs[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest = {
        "schema": "atlas-stage2-manifest-1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "program": VERSION,
        "python": sys.version.split()[0],
        "input_json_gz_count": len(files),
        "document_count": len(documents),
        "feo_measurement_count": len(feo),
        "feo_review_count": sum(row["needs_review"] == "yes" for row in feo),
        "narrative_page_count": len(narratives),
        "review_queue_count": len(reviews),
        "errors": errors,
        "outputs": outputs,
        "notes": [
            "Blank or absent FeO cells are not converted to zero.",
            "Only coordinate-recovered numeric FeO cells appear in atlas_feo_measurements.csv.",
            "Rows marked needs_review=yes must not be treated as final ground truth.",
            "Narrative pages exclude pages beginning at an explicit References/Bibliography heading.",
        ],
    }
    atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Documents: {len(documents)}")
    print(f"FeO measurements: {len(feo)} ({manifest['feo_review_count']} need review)")
    print(f"Narrative pages: {len(narratives)}")
    print(f"Review queue: {len(reviews)}")
    print(f"Errors: {len(errors)}")
    print(f"Manifest: {manifest_path}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
