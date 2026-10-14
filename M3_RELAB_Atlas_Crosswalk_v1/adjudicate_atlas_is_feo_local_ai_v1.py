#!/usr/bin/env python3
"""Adjudicate Atlas I_s/FeO text chunks with a local llama.cpp/Qwen server.

The deterministic collector must run first.  This program sends one short
Atlas page chunk at a time to a server bound to localhost, checkpoints every
successful response, and exports a flat CSV.  Every extracted association is
left as ``needs_human_review``; the model output is never treated as confirmed.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_API_URL = "http://127.0.0.1:8080/v1/chat/completions"
NUMBER_RE = re.compile(r"[<>~≈]?\s*\d+(?:\.\d+)?")

CSV_COLUMNS = [
    "chunk_id",
    "document_id",
    "pdf_page",
    "assigned_sample_id",
    "value_text",
    "qualifier",
    "scope",
    "evidence_quote",
    "association_explanation",
    "source_url",
    "source_pdf_sha256",
    "review_status",
    "validation_status",
    "validation_warnings",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use local Qwen to map Atlas I_s/FeO statements to sample IDs."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="M3_RELAB_Atlas_Crosswalk_v1 directory.",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=None,
        help="Input JSONL; defaults to data/is_feo_v1/atlas_is_feo_review_chunks.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to data/is_feo_v1/local_ai.",
    )
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--model", default="local-qwen")
    parser.add_argument("--max-output-tokens", type=int, default=700)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most this many not-yet-completed chunks; 0 means all.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Request a non-streamed response (recommended for unattended runs).",
    )
    parser.add_argument(
        "--allow-thinking",
        action="store_true",
        help="Allow Qwen thinking mode; disabled by default.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Discard the prior checkpoint before starting.",
    )
    return parser.parse_args()


def clean_space(value: str) -> str:
    return " ".join(str(value).replace("\x00", " ").replace("\xa0", " ").split())


def normalize_for_check(value: str) -> str:
    return re.sub(r"[^a-z0-9<>~=≈.]+", " ", clean_space(value).lower()).strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object on {path}:{line_number}")
            rows.append(value)
    return rows


def build_messages(chunk: dict[str, Any], disable_thinking: bool) -> list[dict[str, str]]:
    system = (
        "You extract lunar-sample I_s/FeO maturity-index statements from supplied "
        "text. Use only the evidence. Return JSON only. Do not use prior knowledge. "
        "Do not read or estimate a value from a graph, axis, or plotted point. Do "
        "not transfer a value between a parent soil and a child, split, depth, grain-"
        "size fraction, rock, breccia, or glass bead. A list using 'respectively' may "
        "be paired only when the text explicitly gives equal-length ordered lists. "
        "Keep ambiguous associations unresolved. Bhatt is not an allowed source."
    )
    switch = "/no_think\n" if disable_thinking else ""
    user = f"""{switch}Chunk ID: {chunk['chunk_id']}
Atlas document sample IDs: {json.dumps(chunk.get('atlas_document_sample_ids', []))}
PDF page: {chunk.get('pdf_page')}

Return exactly this JSON shape:
{{
  "chunk_id": "{chunk['chunk_id']}",
  "measurements": [
    {{
      "sample_id": "12345",
      "value_text": "48",
      "qualifier": "",
      "scope": "soil, split, depth, grain-size fraction, or other scope stated",
      "evidence_quote": "short exact quote containing the association",
      "association_explanation": "why this exact value belongs to this exact sample"
    }}
  ],
  "unresolved_statements": ["ambiguous statement copied or briefly described"],
  "warnings": []
}}

Rules:
- Extract only numerical I_s/FeO values explicitly printed in prose or a table.
- Preserve <, >, ~, or a range in value_text/qualifier; do not turn it into equality.
- Sample IDs must retain any printed comma-split suffix when it identifies a split.
- Qualitative words alone (mature, submature, immature) are not measurements.
- If the text says several samples and values "respectively", pair them in order.
- If a graph is mentioned but no numerical value is printed in prose/table text,
  return no measurement for it.
- If scope or sample/value association is uncertain, put it in unresolved_statements.

EVIDENCE:
{chunk.get('text', '')}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_model_json(raw: str) -> dict[str, Any]:
    candidate = raw.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.I)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(candidate[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Model response is not a JSON object")
    return value


def call_model(
    api_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: int,
    stream: bool,
    disable_thinking: bool,
) -> tuple[str, dict[str, Any]]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    request = Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        if not stream:
            body = json.loads(response.read().decode("utf-8"))
            message = body["choices"][0]["message"]
            return (message.get("content") or "").strip(), body.get("usage", {})

        parts: list[str] = []
        usage: dict[str, Any] = {}
        for byte_line in response:
            line = byte_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            choices = event.get("choices") or []
            if choices:
                parts.append((choices[0].get("delta") or {}).get("content") or "")
        return "".join(parts).strip(), usage


def validate_measurement(row: dict[str, Any], chunk: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    text_normalized = normalize_for_check(chunk.get("text", ""))
    sample_id = clean_space(row.get("sample_id", ""))
    value_text = clean_space(row.get("value_text", ""))
    quote = clean_space(row.get("evidence_quote", ""))
    if not sample_id:
        warnings.append("missing_sample_id")
    elif normalize_for_check(sample_id) not in text_normalized:
        warnings.append("sample_id_not_found_in_chunk")
    if not value_text or not NUMBER_RE.search(value_text):
        warnings.append("missing_numeric_value")
    else:
        number = NUMBER_RE.search(value_text)
        assert number is not None
        numeric_core = re.sub(r"[^0-9.]", "", number.group(0))
        if numeric_core and numeric_core not in text_normalized:
            warnings.append("numeric_value_not_found_in_chunk")
    if not quote:
        warnings.append("missing_evidence_quote")
    elif normalize_for_check(quote) not in text_normalized:
        warnings.append("evidence_quote_not_verbatim_after_normalization")
    return warnings


def flatten_results(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for record in records:
        chunk = record["chunk"]
        result = record["result"]
        for measurement in result.get("measurements", []):
            if not isinstance(measurement, dict):
                continue
            warnings = validate_measurement(measurement, chunk)
            rows.append(
                {
                    "chunk_id": chunk.get("chunk_id", ""),
                    "document_id": chunk.get("document_id", ""),
                    "pdf_page": chunk.get("pdf_page", ""),
                    "assigned_sample_id": measurement.get("sample_id", ""),
                    "value_text": measurement.get("value_text", ""),
                    "qualifier": measurement.get("qualifier", ""),
                    "scope": measurement.get("scope", ""),
                    "evidence_quote": measurement.get("evidence_quote", ""),
                    "association_explanation": measurement.get(
                        "association_explanation", ""
                    ),
                    "source_url": chunk.get("source_url", ""),
                    "source_pdf_sha256": chunk.get("source_pdf_sha256", ""),
                    "review_status": "needs_human_review",
                    "validation_status": "pass" if not warnings else "warning",
                    "validation_warnings": ";".join(warnings),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    temporary = path.with_name(path.name + ".part")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    chunks_path = (
        args.chunks.expanduser().resolve()
        if args.chunks
        else project_root / "data/is_feo_v1/atlas_is_feo_review_chunks.jsonl"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else project_root / "data/is_feo_v1/local_ai"
    )
    if not chunks_path.is_file():
        print(f"ERROR: chunk file not found: {chunks_path}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "atlas_is_feo_ai_checkpoint.jsonl"
    errors_path = output_dir / "atlas_is_feo_ai_errors.jsonl"
    csv_path = output_dir / "atlas_is_feo_ai_measurements.csv"
    manifest_path = output_dir / "atlas_is_feo_ai_manifest.json"
    if args.restart:
        checkpoint.unlink(missing_ok=True)
        errors_path.unlink(missing_ok=True)

    try:
        chunks = read_jsonl(chunks_path)
        checkpoint_records = read_jsonl(checkpoint) if checkpoint.is_file() else []
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    chunks_by_id = {str(row.get("chunk_id", "")): row for row in chunks}
    completed_by_id: dict[str, dict[str, Any]] = {}
    for record in checkpoint_records:
        stored = record.get("chunk", {})
        chunk_id = str(stored.get("chunk_id", ""))
        current = chunks_by_id.get(chunk_id)
        if current is None:
            continue
        # Do not silently reuse an answer if the chopped evidence changed.
        if stored.get("chunk_text_sha256") != current.get("chunk_text_sha256"):
            continue
        completed_by_id[chunk_id] = record
    completed_ids = set(completed_by_id)
    pending = [row for row in chunks if str(row.get("chunk_id", "")) not in completed_ids]
    if args.limit > 0:
        pending = pending[: args.limit]

    failures = 0
    for number, chunk in enumerate(pending, 1):
        chunk_id = str(chunk.get("chunk_id", ""))
        print(f"[{number}/{len(pending)}] {chunk_id}", flush=True)
        try:
            raw, usage = call_model(
                api_url=args.api_url,
                model=args.model,
                messages=build_messages(chunk, disable_thinking=not args.allow_thinking),
                temperature=args.temperature,
                max_tokens=args.max_output_tokens,
                timeout=args.timeout,
                stream=not args.no_stream,
                disable_thinking=not args.allow_thinking,
            )
            if not raw:
                raise ValueError("model returned empty content")
            result = parse_model_json(raw)
            if result.get("chunk_id") != chunk_id:
                raise ValueError("model returned a different chunk_id")
            if not isinstance(result.get("measurements"), list):
                raise ValueError("model measurements is not a list")
            record = {
                "schema": "atlas-is-feo-local-ai-result-1.0",
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "chunk": chunk,
                "result": result,
                "raw_model_response": raw,
                "model_usage": usage,
                "model_request_name": args.model,
                "temperature": args.temperature,
            }
            append_jsonl(checkpoint, record)
            completed_by_id[chunk_id] = record
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
            failures += 1
            append_jsonl(
                errors_path,
                {
                    "failed_utc": datetime.now(timezone.utc).isoformat(),
                    "chunk_id": chunk_id,
                    "error": str(exc),
                },
            )
            print(f"  ERROR: {exc}", file=sys.stderr)

    completed = [
        completed_by_id[str(chunk.get("chunk_id", ""))]
        for chunk in chunks
        if str(chunk.get("chunk_id", "")) in completed_by_id
    ]
    rows = flatten_results(completed)
    write_csv(csv_path, rows)
    manifest = {
        "schema": "atlas-is-feo-local-ai-manifest-1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "chunks_file": str(chunks_path),
        "checkpoint_file": str(checkpoint),
        "measurements_csv": str(csv_path),
        "errors_file": str(errors_path),
        "model_request_name": args.model,
        "api_url": args.api_url,
        "temperature": args.temperature,
        "thinking_requested": args.allow_thinking,
        "rules": {
            "all_measurements_need_human_review": True,
            "parent_child_transfer": "forbidden",
            "graph_reading": "forbidden",
            "bhatt_allowed": False,
        },
        "counts": {
            "input_chunks": len(chunks),
            "completed_chunks": len(completed),
            "pending_chunks": len(chunks) - len(completed),
            "measurement_rows": len(rows),
            "failures_this_run": failures,
            "validation_warning_rows": sum(
                row["validation_status"] == "warning" for row in rows
            ),
        },
    }
    temporary = manifest_path.with_name(manifest_path.name + ".part")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    print(f"Completed chunks: {len(completed)}/{len(chunks)}")
    print(f"Candidate measurement rows: {len(rows)}")
    print(f"CSV: {csv_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
