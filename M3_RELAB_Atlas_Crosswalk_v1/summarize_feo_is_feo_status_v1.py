#!/usr/bin/env python3
"""Summarize confirmed/candidate/unresolved FeO and I_s/FeO work.

The report deliberately keeps three classes separate:

* accepted FeO decisions from the reviewed FeO pipeline;
* previously recorded I_s/FeO measurements;
* Qwen-produced I_s/FeO candidates, all of which still need human review.

Missing files are reported rather than treated as empty completed work.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="M3_RELAB_Atlas_Crosswalk_v1 directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to data/is_feo_v1.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if isinstance(value, dict):
                rows.append(value)
    return rows


def group(rows: Iterable[dict[str, str]], column: str) -> dict[str, int]:
    counts = Counter((row.get(column, "") or "<blank>").strip() for row in rows)
    return dict(sorted(counts.items()))


def file_state(path: Path, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "present": path.is_file()}
    if rows is not None:
        result["rows"] = rows
    return result


def summarize(root: Path, output_dir: Path) -> dict[str, Any]:
    ai_dir = root / "data/ai_resolution_v1"
    parent_summary_path = ai_dir / "review_parent_summary.csv"
    adjudications_path = ai_dir / "review_candidate_adjudications.csv"
    missing_path = ai_dir / "missing_case_remaining.csv"

    parent_rows = read_csv(parent_summary_path)
    adjudication_rows = read_csv(adjudications_path)
    missing_rows = read_csv(missing_path)

    existing_path = output_dir / "existing_is_feo_records.csv"
    relab_path = output_dir / "relab_is_feo_hits.csv"
    atlas_path = output_dir / "atlas_is_feo_candidates.csv"
    chunks_path = output_dir / "atlas_is_feo_review_chunks.jsonl"
    checkpoint_path = output_dir / "local_ai/atlas_is_feo_ai_checkpoint.jsonl"
    measurements_path = output_dir / "local_ai/atlas_is_feo_ai_measurements.csv"
    errors_path = output_dir / "local_ai/atlas_is_feo_ai_errors.jsonl"

    existing_rows = read_csv(existing_path)
    relab_rows = read_csv(relab_path)
    atlas_rows = read_csv(atlas_path)
    chunks = read_jsonl(chunks_path)
    checkpoint_rows = read_jsonl(checkpoint_path)
    measurement_rows = read_csv(measurements_path)
    error_rows = read_jsonl(errors_path)

    chunks_by_id = {
        str(row.get("chunk_id", "")): row for row in chunks if row.get("chunk_id")
    }
    chunk_ids = set(chunks_by_id)
    completed_by_id: dict[str, dict[str, Any]] = {}
    for record in checkpoint_rows:
        stored = record.get("chunk") or {}
        chunk_id = str(stored.get("chunk_id", ""))
        current = chunks_by_id.get(chunk_id)
        if current is None:
            continue
        if stored.get("chunk_text_sha256") != current.get("chunk_text_sha256"):
            continue
        completed_by_id[chunk_id] = record

    unresolved_statement_count = 0
    chunks_with_unresolved = 0
    model_warning_count = 0
    for record in completed_by_id.values():
        result = record.get("result") or {}
        unresolved = result.get("unresolved_statements") or []
        if isinstance(unresolved, list) and unresolved:
            chunks_with_unresolved += 1
            unresolved_statement_count += len(unresolved)
        warnings = result.get("warnings") or []
        if isinstance(warnings, list):
            model_warning_count += len(warnings)

    feo_status = group(parent_rows, "resolution_status")
    resolved_parent_rows = sum(
        count for status, count in feo_status.items() if status.startswith("resolved_")
    )
    unresolved_parent_rows = len(parent_rows) - resolved_parent_rows

    return {
        "schema": "feo-is-feo-status-1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rules": {
            "bhatt_allowed": False,
            "parent_child_transfer": "forbidden",
            "graph_reading": "forbidden",
            "qwen_rows_are_confirmed": False,
            "is_definition": (
                "calibrated ferromagnetic-resonance intensity near g=2.1; "
                "not a spectral-reflectance ratio"
            ),
        },
        "feo": {
            "reviewed_parent_rows": len(parent_rows),
            "resolved_parent_rows": resolved_parent_rows,
            "unresolved_or_no_parent_bulk_rows": unresolved_parent_rows,
            "parent_resolution_status": feo_status,
            "candidate_adjudication_rows": len(adjudication_rows),
            "candidate_decisions": group(adjudication_rows, "decision"),
            "remaining_priority_cases": len(missing_rows),
            "remaining_priority_sample_ids": [
                row.get("apollo_parent_id", "") for row in missing_rows
            ],
            "files": {
                "parent_summary": file_state(parent_summary_path, len(parent_rows)),
                "adjudications": file_state(adjudications_path, len(adjudication_rows)),
                "remaining": file_state(missing_path, len(missing_rows)),
            },
        },
        "is_feo": {
            "previously_recorded_rows": len(existing_rows),
            "previously_recorded_review_flags": group(existing_rows, "needs_review"),
            "relab_metadata_hits": len(relab_rows),
            "relab_hit_kinds": group(relab_rows, "match_kind"),
            "atlas_candidate_pages": len(atlas_rows),
            "atlas_candidate_kinds": group(atlas_rows, "candidate_kind"),
            "qwen_input_chunks": len(chunks),
            "qwen_completed_chunks": len(completed_by_id),
            "qwen_pending_chunks": len(chunk_ids - set(completed_by_id)),
            "qwen_error_rows": len(error_rows),
            "qwen_candidate_measurement_rows": len(measurement_rows),
            "qwen_validation_status": group(measurement_rows, "validation_status"),
            "qwen_review_status": group(measurement_rows, "review_status"),
            "qwen_chunks_with_unresolved_statements": chunks_with_unresolved,
            "qwen_unresolved_statement_count": unresolved_statement_count,
            "qwen_model_warning_count": model_warning_count,
            "files": {
                "existing": file_state(existing_path, len(existing_rows)),
                "relab_hits": file_state(relab_path, len(relab_rows)),
                "atlas_candidates": file_state(atlas_path, len(atlas_rows)),
                "chunks": file_state(chunks_path, len(chunks)),
                "checkpoint": file_state(checkpoint_path, len(checkpoint_rows)),
                "measurements": file_state(measurements_path, len(measurement_rows)),
                "errors": file_state(errors_path, len(error_rows)),
            },
        },
    }


def markdown(report: dict[str, Any]) -> str:
    feo = report["feo"]
    maturity = report["is_feo"]
    sample_ids = ", ".join(feo["remaining_priority_sample_ids"]) or "none"
    return f"""# FeO and I_s/FeO status

Generated: {report['created_utc']}

## Interpretation rules

- Bhatt-derived material: excluded.
- Parent-to-child transfer: forbidden.
- Graph/curve reading: forbidden.
- Qwen output: candidates requiring human review, never confirmed truth.
- I_s: calibrated ferromagnetic-resonance intensity near g=2.1, not a spectral-reflectance ratio.

## FeO reviewed pipeline

- Reviewed parent rows: {feo['reviewed_parent_rows']}
- Resolved parent rows: {feo['resolved_parent_rows']}
- Unresolved or no defensible parent-bulk rows: {feo['unresolved_or_no_parent_bulk_rows']}
- Remaining priority cases: {feo['remaining_priority_cases']} ({sample_ids})
- Candidate adjudication rows: {feo['candidate_adjudication_rows']}
- Resolution statuses: {json.dumps(feo['parent_resolution_status'], sort_keys=True)}
- Candidate decisions: {json.dumps(feo['candidate_decisions'], sort_keys=True)}

## I_s/FeO recovery

- Previously recorded rows: {maturity['previously_recorded_rows']}
- RELAB metadata hits: {maturity['relab_metadata_hits']}
- Atlas candidate pages: {maturity['atlas_candidate_pages']}
- Qwen input chunks: {maturity['qwen_input_chunks']}
- Qwen completed / pending: {maturity['qwen_completed_chunks']} / {maturity['qwen_pending_chunks']}
- Qwen errors: {maturity['qwen_error_rows']}
- Qwen candidate measurement rows: {maturity['qwen_candidate_measurement_rows']}
- Chunks with unresolved statements: {maturity['qwen_chunks_with_unresolved_statements']}
- Total unresolved statements: {maturity['qwen_unresolved_statement_count']}
- Validation statuses: {json.dumps(maturity['qwen_validation_status'], sort_keys=True)}

Missing inputs remain marked `present: false` in the JSON report; they are not counted as completed work.
"""


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else root / "data/is_feo_v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        report = summarize(root, output_dir)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    json_path = output_dir / "feo_is_feo_status.json"
    md_path = output_dir / "FEO_IS_FEO_STATUS.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(markdown(report), encoding="utf-8")
    print(markdown(report))
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
