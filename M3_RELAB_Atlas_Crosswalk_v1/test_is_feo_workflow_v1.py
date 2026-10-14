#!/usr/bin/env python3

import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

import adjudicate_atlas_is_feo_local_ai_v1 as ai
import collect_is_feo_candidates_v1 as collect


class IsFeoWorkflowTests(unittest.TestCase):
    def test_spelling_variants(self) -> None:
        for text in ("Is/FeO = 48", "I_s / FeO = 48", "I sub s per FeO = 48", "Iₛ/FeO = 48"):
            self.assertIsNotNone(collect.IS_FEO_RE.search(text), text)

    def test_candidate_classification(self) -> None:
        statement = "The maturity of 76501 is Is/FeO = 58 (Morris 1978)."
        citation = (
            "Morris R.V. (1978) The surface exposure (maturity) of lunar soils: "
            "Some concepts and Is/FeO compilation. Proc. 9th Lunar Sci. Conf."
        )
        graph = "0 20 40 60 80 100 120 maturity (Is/FeO) Figure 4"
        self.assertEqual(
            collect.atlas_candidate_kind(
                statement, collect.unique_join(collect.numeric_tokens_near_is_feo(statement))
            ),
            "numeric_statement",
        )
        self.assertEqual(
            collect.atlas_candidate_kind(
                citation, collect.unique_join(collect.numeric_tokens_near_is_feo(citation))
            ),
            "citation_only",
        )
        self.assertEqual(
            collect.atlas_candidate_kind(
                graph, collect.unique_join(collect.numeric_tokens_near_is_feo(graph))
            ),
            "numeric_graph_or_ambiguous",
        )

    def test_atlas_scan_keeps_provenance_and_excludes_bhatt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = {
                "source": {
                    "document_id": "76501__compendium__test",
                    "atlas_sample_ids": ["76501"],
                    "source_url": "https://example.test/76501.pdf",
                    "source_pdf_sha256": "abc123",
                },
                "pages": [
                    {
                        "page_number": 2,
                        "text": "Petrography\nThe maturity of 76501 is Is/FeO = 58 (Morris 1978).",
                    },
                    {
                        "page_number": 3,
                        "text": "Bhatt reports Is/FeO = 99 for 76501.",
                    },
                ],
            }
            path = root / "76501.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                json.dump(document, handle)
            rows, chunks, failures = collect.scan_atlas_json(
                root, 2, collect.re.compile(collect.EXCLUDED_DEFAULT, collect.re.I)
            )
            self.assertEqual(failures, 0)
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(chunks), 1)
            self.assertEqual(rows[0]["source_pdf_sha256"], "abc123")
            self.assertEqual(rows[0]["raw_numeric_tokens"], "58")

    def test_model_measurement_validation(self) -> None:
        chunk = {"text": "The maturity of 76501 is Is/FeO = 58."}
        good = {
            "sample_id": "76501",
            "value_text": "58",
            "evidence_quote": "The maturity of 76501 is Is/FeO = 58.",
        }
        bad = {
            "sample_id": "76502",
            "value_text": "99",
            "evidence_quote": "invented",
        }
        self.assertEqual(ai.validate_measurement(good, chunk), [])
        warnings = ai.validate_measurement(bad, chunk)
        self.assertIn("sample_id_not_found_in_chunk", warnings)
        self.assertIn("numeric_value_not_found_in_chunk", warnings)
        self.assertIn("evidence_quote_not_verbatim_after_normalization", warnings)


if __name__ == "__main__":
    unittest.main()
