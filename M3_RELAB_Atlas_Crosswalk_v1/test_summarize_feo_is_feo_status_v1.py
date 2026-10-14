import csv
import json
import tempfile
import unittest
from pathlib import Path

import summarize_feo_is_feo_status_v1 as summary


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class StatusSummaryTests(unittest.TestCase):
    def test_separates_completed_pending_and_unresolved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ai = root / "data/ai_resolution_v1"
            out = root / "data/is_feo_v1"
            write_csv(
                ai / "review_parent_summary.csv",
                ["resolution_status"],
                [
                    {"resolution_status": "resolved_with_parent_feo"},
                    {"resolution_status": "reviewed_no_parent_bulk_feo"},
                ],
            )
            write_csv(
                ai / "review_candidate_adjudications.csv",
                ["decision"],
                [{"decision": "accepted"}, {"decision": "rejected"}],
            )
            write_csv(
                ai / "missing_case_remaining.csv",
                ["apollo_parent_id"],
                [{"apollo_parent_id": "60215"}],
            )
            write_jsonl(
                out / "atlas_is_feo_review_chunks.jsonl",
                [
                    {"chunk_id": "one", "chunk_text_sha256": "current-one"},
                    {"chunk_id": "two", "chunk_text_sha256": "current-two"},
                ],
            )
            write_jsonl(
                out / "local_ai/atlas_is_feo_ai_checkpoint.jsonl",
                [
                    {
                        "chunk": {
                            "chunk_id": "one",
                            "chunk_text_sha256": "current-one",
                        },
                        "result": {
                            "unresolved_statements": ["ambiguous"],
                            "warnings": [],
                        },
                    }
                ],
            )
            report = summary.summarize(root, out)
            self.assertEqual(report["feo"]["resolved_parent_rows"], 1)
            self.assertEqual(report["feo"]["unresolved_or_no_parent_bulk_rows"], 1)
            self.assertEqual(report["is_feo"]["qwen_completed_chunks"], 1)
            self.assertEqual(report["is_feo"]["qwen_pending_chunks"], 1)
            self.assertEqual(report["is_feo"]["qwen_unresolved_statement_count"], 1)

    def test_stale_checkpoint_is_pending(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out = root / "data/is_feo_v1"
            write_jsonl(
                out / "atlas_is_feo_review_chunks.jsonl",
                [{"chunk_id": "one", "chunk_text_sha256": "new"}],
            )
            write_jsonl(
                out / "local_ai/atlas_is_feo_ai_checkpoint.jsonl",
                [
                    {
                        "chunk": {"chunk_id": "one", "chunk_text_sha256": "old"},
                        "result": {"unresolved_statements": ["stale"]},
                    }
                ],
            )
            report = summary.summarize(root, out)
            self.assertEqual(report["is_feo"]["qwen_completed_chunks"], 0)
            self.assertEqual(report["is_feo"]["qwen_pending_chunks"], 1)
            self.assertEqual(report["is_feo"]["qwen_unresolved_statement_count"], 0)


if __name__ == "__main__":
    unittest.main()

