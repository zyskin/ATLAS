#!/usr/bin/env python3
import unittest

import build_atlas_parents_without_feo_v2 as v2


class ParentsWithoutFeOV2Tests(unittest.TestCase):
    def test_sources_are_combined_without_promoting_mineral_phase(self):
        relab = [
            {"i_atlas_sample": "10003", "k_feo_wt_percent": "-99", "feo_known": "no", "relab_measurement_id": "r1"},
            {"i_atlas_sample": "10004", "k_feo_wt_percent": "-99", "feo_known": "no", "relab_measurement_id": "r2"},
            {"i_atlas_sample": "10005", "k_feo_wt_percent": "-99", "feo_known": "no", "relab_measurement_id": "r3"},
            {"i_atlas_sample": "10009", "k_feo_wt_percent": "-99", "feo_known": "no", "relab_measurement_id": "r4"},
            {"i_atlas_sample": "10010", "k_feo_wt_percent": "12.5", "feo_known": "yes", "relab_measurement_id": "r5"},
        ]
        baseline, _ = v2.baseline_without_relab_feo(relab, -99.0)
        self.assertEqual(set(baseline), {10003, 10004, 10005, 10009})

        pdf = [
            {"measurement_id": "1", "document_id": "d3", "atlas_sample_id": "10003", "pdf_page": "6", "value_wt_percent": "19.8", "value_text": "19.8", "sample_scope": "bulk_or_whole_sample", "needs_review": "no"},
            {"measurement_id": "2", "document_id": "d4", "atlas_sample_id": "10004", "pdf_page": "4", "value_wt_percent": "8.2", "value_text": "8.2", "sample_scope": "bulk_or_whole_sample", "needs_review": "yes"},
            {"measurement_id": "3", "document_id": "d5", "atlas_sample_id": "10005", "pdf_page": "3", "value_wt_percent": "5.0", "value_text": "5.0", "sample_scope": "mineral_phase", "needs_review": "no"},
        ]
        grouped = v2.classify_pdf_rows(pdf, baseline)
        all_rows, no_rows = v2.build_outputs(baseline, grouped)
        by_parent = {row["apollo_parent_id"]: row for row in all_rows}

        self.assertEqual(by_parent[10003]["pdf_parent_feo_status"], "accepted_pdf_bulk_feo")
        self.assertEqual(by_parent[10003]["pdf_feo_accepted_bulk_wt_percent"], "19.8")
        self.assertEqual(by_parent[10004]["pdf_parent_feo_status"], "review_only_pdf_parent_feo_candidates")
        self.assertEqual(by_parent[10005]["pdf_parent_feo_status"], "no_parent_level_pdf_feo_candidate")
        self.assertEqual(by_parent[10005]["pdf_feo_mineral_phase_not_parent_wt_percent"], "5.0")
        self.assertEqual([row["apollo_parent_id"] for row in no_rows], [10005, 10009])


if __name__ == "__main__":
    unittest.main()
