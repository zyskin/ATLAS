#!/usr/bin/env python3
"""Extract FeO evidence from one Lunar Sample Atlas PDF using local Qwen.

The script deliberately separates two jobs:

1. Deterministic PDF processing finds pages and short, layout-preserving text
   blocks containing FeO/chemistry terms.
2. One OpenAI-compatible call to a locally running llama.cpp server interprets
   only those blocks and returns structured JSON with page-level evidence.

No PDF text is sent to an external service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pdfplumber


DEFAULT_API_URL = "http://127.0.0.1:8080/v1/chat/completions"
KEYWORD_RE = re.compile(
    r"(?i)(?<![A-Za-z])FeO(?![A-Za-z])|ferrous\s+oxide|iron\s+oxide"
)
CHEMISTRY_RE = re.compile(
    r"(?i)chemical\s+composition|chemistry|composition|weight\s*%|wt\.?\s*%"
)


@dataclass(frozen=True)
class EvidenceBlock:
    page: int
    start_line: int
    end_line: int
    score: int
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find FeO-related passages in one Atlas PDF and optionally ask a "
            "local llama.cpp/Qwen server to convert them to auditable JSON."
        )
    )
    parser.add_argument(
        "pdf",
        help="Local PDF path or an https:// URL to a PDF.",
    )
    parser.add_argument(
        "--sample-id",
        default="",
        help="Apollo sample number, e.g. 10003. Inferred from filename if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("atlas_feo_output"),
        help="Output directory (default: ./atlas_feo_output).",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"Local OpenAI-compatible endpoint (default: {DEFAULT_API_URL}).",
    )
    parser.add_argument(
        "--model",
        default="local-qwen",
        help="Model name sent to llama.cpp; the loaded model is used by default.",
    )
    parser.add_argument(
        "--max-evidence-chars",
        type=int,
        default=9000,
        help="Maximum evidence characters sent to the model (default: 9000).",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=450,
        help="Maximum generated tokens (default: 450).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature (default: 0.0 for extraction).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="HTTP timeout in seconds for the slow local model (default: 3600).",
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Only extract/rank evidence; do not call the local model.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Wait silently for one complete API response instead of showing streamed output.",
    )
    parser.add_argument(
        "--allow-thinking",
        action="store_true",
        help="Allow Qwen thinking mode (disabled by default for extraction).",
    )
    parser.add_argument(
        "--keep-pdf",
        action="store_true",
        help="Keep a downloaded copy of a URL input inside the output directory.",
    )
    return parser.parse_args()


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def safe_sample_id(pdf_arg: str, supplied: str) -> str:
    candidate = supplied.strip() or Path(urlparse(pdf_arg).path).stem
    candidate = re.sub(r"[^A-Za-z0-9_.-]+", "_", candidate).strip("_")
    return candidate or "unknown_sample"


def acquire_pdf(pdf_arg: str, output_dir: Path, keep_pdf: bool) -> tuple[Path, bool]:
    """Return (local_path, is_temporary)."""
    if not is_url(pdf_arg):
        path = Path(pdf_arg).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {path}")
        return path, False

    request = Request(pdf_arg, headers={"User-Agent": "atlas-feo-extractor/1.0"})
    with urlopen(request, timeout=120) as response:
        content = response.read()
    if not content.startswith(b"%PDF"):
        raise ValueError(f"URL did not return a PDF: {pdf_arg}")

    if keep_pdf:
        destination = output_dir / Path(urlparse(pdf_arg).path).name
        destination.write_bytes(content)
        return destination, False

    handle = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    handle.write(content)
    handle.close()
    return Path(handle.name), True


def clean_layout_text(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\x00", "").splitlines()]
    # Retain internal spaces because they carry table-column alignment.
    return "\n".join(lines).strip()


def make_blocks_for_page(page_number: int, text: str) -> list[EvidenceBlock]:
    lines = text.splitlines()
    hits = [i for i, line in enumerate(lines) if KEYWORD_RE.search(line)]
    if not hits:
        return []

    # Chemistry tables are the common case in the Atlas. Sending the whole
    # table is slow and unnecessary. Preserve the spatially aligned title,
    # reference header, FeO row, and technique key, even when they are far
    # apart on the page.
    has_chemistry_table = any(
        re.search(r"(?i)table\s+\S+.*chemical\s+composition", line)
        for line in lines
    ) and any(re.search(r"(?i)^\s*reference\b", line) for line in lines)
    if has_chemistry_table:
        selected_indices = [
            index
            for index, line in enumerate(lines)
            if re.search(r"(?i)table\s+\S+.*chemical\s+composition", line)
            or re.search(r"(?i)^\s*reference\b", line)
            or KEYWORD_RE.search(line)
            or re.search(r"(?i)^\s*technique\s*:", line)
        ]
        block_text = "\n".join(lines[index] for index in selected_indices).strip()
        keyword_count = len(KEYWORD_RE.findall(block_text))
        numeric_count = len(re.findall(r"(?<!\w)\d+(?:\.\d+)?(?!\w)", block_text))
        return [
            EvidenceBlock(
                page=page_number,
                start_line=min(selected_indices) + 1,
                end_line=max(selected_indices) + 1,
                score=40 + 20 * keyword_count + min(numeric_count, 20),
                text=block_text,
            )
        ]

    ranges: list[tuple[int, int]] = []
    for hit in hits:
        # A generous look-back keeps table titles and column/reference headings.
        start = max(0, hit - 16)
        end = min(len(lines), hit + 13)
        if ranges and start <= ranges[-1][1] + 2:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
        else:
            ranges.append((start, end))

    blocks: list[EvidenceBlock] = []
    for start, end in ranges:
        block_text = "\n".join(lines[start:end]).strip()
        keyword_count = len(KEYWORD_RE.findall(block_text))
        chemistry_count = len(CHEMISTRY_RE.findall(block_text))
        numeric_count = len(re.findall(r"(?<!\w)\d+(?:\.\d+)?(?!\w)", block_text))
        score = 20 * keyword_count + 4 * chemistry_count + min(numeric_count, 20)
        blocks.append(
            EvidenceBlock(
                page=page_number,
                start_line=start + 1,
                end_line=end,
                score=score,
                text=block_text,
            )
        )
    return blocks


def extract_evidence(pdf_path: Path) -> tuple[list[EvidenceBlock], dict[str, Any]]:
    blocks: list[EvidenceBlock] = []
    pages_with_text = 0
    page_count = 0

    with pdfplumber.open(pdf_path) as pdf:
        page_count = len(pdf.pages)
        for page_number, page in enumerate(pdf.pages, 1):
            # layout=True keeps approximate columns, essential for chemistry tables.
            text = clean_layout_text(page.extract_text(layout=True) or "")
            if text:
                pages_with_text += 1
            blocks.extend(make_blocks_for_page(page_number, text))

    blocks.sort(key=lambda block: (-block.score, block.page, block.start_line))
    metadata = {
        "pdf_pages": page_count,
        "pages_with_extractable_text": pages_with_text,
        "candidate_blocks": len(blocks),
    }
    return blocks, metadata


def evidence_document(blocks: list[EvidenceBlock], max_chars: int) -> str:
    selected: list[str] = []
    used = 0
    for number, block in enumerate(blocks, 1):
        header = (
            f"=== EVIDENCE BLOCK {number}: PDF page {block.page}, "
            f"extracted lines {block.start_line}-{block.end_line}, "
            f"ranking score {block.score} ===\n"
        )
        item = header + block.text + "\n"
        if selected and used + len(item) > max_chars:
            continue
        if not selected and len(item) > max_chars:
            item = item[:max_chars] + "\n[TRUNCATED]\n"
        selected.append(item)
        used += len(item)
        if used >= max_chars:
            break
    return "\n".join(selected)


def build_prompt(
    sample_id: str, evidence: str, disable_thinking: bool
) -> list[dict[str, str]]:
    system = (
        "You extract lunar-sample chemistry from supplied evidence. "
        "Use only the evidence. Do not use prior knowledge. Do not infer a number "
        "from a graph. Preserve every independently reported FeO value. A blank "
        "table cell is not zero. If plain-text table alignment does not securely "
        "associate a value with a reference, keep the source_reference null and "
        "set table_alignment_uncertain true. Return JSON only, without markdown "
        "or reasoning text."
    )
    mode_switch = "/no_think\n" if disable_thinking else ""
    user = f"""{mode_switch}Apollo sample: {sample_id}

Extract FeO measurements from the evidence below. FeO values are weight percent
unless the evidence explicitly says otherwise. Return exactly this JSON shape:
{{
  "sample_id": "{sample_id}",
  "feo_measurements": [
    {{
      "value_wt_percent": 0.0,
      "source_reference": null,
      "analytical_method": null,
      "pdf_page": 1,
      "evidence_quote": "short exact evidence",
      "table_alignment_uncertain": false,
      "notes": null
    }}
  ],
  "no_numeric_feo_pages": [],
  "warnings": []
}}

Rules:
- Record all explicitly printed numeric FeO values, including repeated values.
- Do not average values and do not select a preferred value.
- Distinguish a reported value from an uncertainty or a range.
- Keep evidence_quote short and verbatim.
- Page numbers must come from EVIDENCE BLOCK labels.
- If an FeO row is present but contains no numeric value, add its page to
  no_numeric_feo_pages; do not create a measurement.

EVIDENCE:
{evidence}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_local_model(
    api_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_output_tokens: int,
    timeout: int,
    stream: bool,
    disable_thinking: bool,
) -> tuple[str, dict[str, Any], str]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_output_tokens,
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
            raw = (message.get("content") or "").strip()
            reasoning = (message.get("reasoning_content") or "").strip()
            return raw, body.get("usage", {}), reasoning

        print("Qwen response (streaming):", flush=True)
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage: dict[str, Any] = {}
        reasoning_chunks = 0
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
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            reasoning_piece = delta.get("reasoning_content") or ""
            if reasoning_piece:
                reasoning_parts.append(reasoning_piece)
                reasoning_chunks += 1
                if reasoning_chunks % 20 == 0:
                    print(".", end="", file=sys.stderr, flush=True)
            content_piece = delta.get("content") or ""
            if content_piece:
                content_parts.append(content_piece)
                print(content_piece, end="", flush=True)
        print(flush=True)
        return (
            "".join(content_parts).strip(),
            usage,
            "".join(reasoning_parts).strip(),
        )


def parse_json_response(raw: str) -> dict[str, Any]:
    candidate = raw.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.I)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(candidate[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Model response JSON is not an object")
    return parsed


def validate_result(result: dict[str, Any], sample_id: str) -> list[str]:
    warnings: list[str] = []
    if str(result.get("sample_id", "")) != sample_id:
        warnings.append("Model returned a different sample_id")
    measurements = result.get("feo_measurements")
    if not isinstance(measurements, list):
        warnings.append("feo_measurements is not a list")
        return warnings
    for index, row in enumerate(measurements, 1):
        if not isinstance(row, dict):
            warnings.append(f"Measurement {index} is not an object")
            continue
        value = row.get("value_wt_percent")
        if not isinstance(value, (int, float)):
            warnings.append(f"Measurement {index} has a nonnumeric FeO value")
        elif not 0 <= float(value) <= 100:
            warnings.append(f"Measurement {index} FeO value is outside 0-100 wt.%")
        if not isinstance(row.get("pdf_page"), int):
            warnings.append(f"Measurement {index} has no integer PDF page")
    return warnings


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_id = safe_sample_id(args.pdf, args.sample_id)
    pdf_path: Path | None = None
    temporary_pdf = False

    try:
        pdf_path, temporary_pdf = acquire_pdf(args.pdf, args.output_dir, args.keep_pdf)
        blocks, pdf_metadata = extract_evidence(pdf_path)
        if pdf_metadata["pages_with_extractable_text"] == 0:
            raise RuntimeError(
                "The PDF has no extractable text. OCR is required before this script can proceed."
            )
        if not blocks:
            raise RuntimeError("No FeO-related text was found in the PDF")

        evidence = evidence_document(blocks, args.max_evidence_chars)
        evidence_path = args.output_dir / f"{sample_id}_feo_evidence.txt"
        evidence_path.write_text(evidence, encoding="utf-8")

        manifest: dict[str, Any] = {
            "schema": "atlas-feo-local-ai-1.0",
            "sample_id": sample_id,
            "source": args.pdf,
            "local_pdf_sha256": sha256_file(pdf_path),
            "pdf": pdf_metadata,
            "evidence_file": evidence_path.name,
            "evidence_characters": len(evidence),
            "ai_called": not args.no_ai,
            "api_url": args.api_url if not args.no_ai else None,
            "model_request_name": args.model if not args.no_ai else None,
            "temperature": args.temperature if not args.no_ai else None,
            "thinking_requested": args.allow_thinking if not args.no_ai else None,
        }

        if args.no_ai:
            manifest_path = args.output_dir / f"{sample_id}_feo_manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"Evidence written: {evidence_path}")
            print("AI was not called (--no-ai).")
            return 0

        disable_thinking = not args.allow_thinking
        messages = build_prompt(sample_id, evidence, disable_thinking)
        raw, usage, reasoning = call_local_model(
            api_url=args.api_url,
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            timeout=args.timeout,
            stream=not args.no_stream,
            disable_thinking=disable_thinking,
        )
        raw_path = args.output_dir / f"{sample_id}_feo_model_raw.txt"
        raw_path.write_text(raw, encoding="utf-8")
        reasoning_path: Path | None = None
        if reasoning:
            reasoning_path = args.output_dir / f"{sample_id}_feo_model_reasoning.txt"
            reasoning_path.write_text(reasoning, encoding="utf-8")
        if not raw:
            raise RuntimeError(
                "Qwen returned no answer content and generated reasoning only. "
                "Inspect the reasoning file and verify that this llama.cpp/Qwen "
                "build honors enable_thinking=false."
            )

        result = parse_json_response(raw)
        validation_warnings = validate_result(result, sample_id)
        result["pipeline_validation_warnings"] = validation_warnings
        result["provenance"] = {
            "source_pdf": args.pdf,
            "source_pdf_sha256": manifest["local_pdf_sha256"],
            "evidence_file": evidence_path.name,
            "raw_model_response_file": raw_path.name,
            "model_usage": usage,
            "temperature": args.temperature,
        }
        result_path = args.output_dir / f"{sample_id}_feo_extraction.json"
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        manifest["result_file"] = result_path.name
        manifest["raw_model_response_file"] = raw_path.name
        manifest["model_reasoning_file"] = (
            reasoning_path.name if reasoning_path is not None else None
        )
        manifest["model_usage"] = usage
        manifest["validation_warnings"] = validation_warnings
        manifest_path = args.output_dir / f"{sample_id}_feo_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        print(f"Evidence: {evidence_path}")
        print(f"Extraction: {result_path}")
        print(f"Raw model response: {raw_path}")
        if reasoning_path is not None:
            print(f"Model reasoning: {reasoning_path}")
        print(f"Manifest: {manifest_path}")
        if validation_warnings:
            print("Validation warnings:", file=sys.stderr)
            for warning in validation_warnings:
                print(f"  - {warning}", file=sys.stderr)
        return 0

    except (HTTPError, URLError, TimeoutError) as exc:
        print(f"ERROR: network/API request failed: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        if temporary_pdf and pdf_path is not None:
            try:
                pdf_path.unlink(missing_ok=True)
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
