#!/usr/bin/env python3
import unittest

import extract_atlas_stage2 as stage2


def word(text, x0, x1, y0, line=0, word_number=0):
    return {
        "text": text, "x0": x0, "x1": x1, "y0": y0, "y1": y0 + 9,
        "block": 1, "line": line, "word": word_number,
    }


class AtlasStage2Tests(unittest.TestCase):
    def test_coordinate_mapping_uses_printed_columns(self):
        words = [
            word("Table", 54, 82, 53), word("1a.", 85, 100, 53),
            word("Chemical", 107, 156, 53), word("composition", 159, 224, 53),
            word("of", 228, 238, 53), word("10003.", 241, 275, 53),
            word("reference", 54, 88, 80),
            word("Gast70", 96, 123, 80), word("Wiesmann75", 167, 215, 80),
            word("Compston70", 241, 288, 80), word("Rose70", 293, 322, 80),
            word("Goles70", 335, 366, 80), word("Rhodes80", 378, 416, 80),
            word("FeO", 54, 70, 128), word("19.8", 241, 258, 128),
            word("(c)", 277, 289, 128), word("19.8", 293, 310, 128),
            word("(d)", 320, 330, 128), word("19.7", 335, 351, 128),
            word("(e)", 363, 373, 128), word("20.51", 378, 399, 128),
            word("(c)", 410, 422, 128),
        ]
        document = {
            "source": {
                "document_id": "doc", "atlas_sample_ids": ["10003"],
                "source_url": "url", "source_pdf_sha256": "sha",
            }
        }
        page = {
            "page_number": 6, "word_coordinates": words,
            "chemistry_layout": {
                "reference_rows": [{"y0": 80, "text": "reference"}],
                "feo_rows": [{"y0": 128, "text": "FeO"}],
                "technique_rows": [{"text": "technique: (c) XRF, (d) semi micro XRF, (e) INAA"}],
            },
        }
        rows, reviews = stage2.extract_feo_from_page(document, page)
        self.assertEqual([r["reference"] for r in rows],
                         ["Compston70", "Rose70", "Goles70", "Rhodes80"])
        self.assertEqual([r["value_wt_percent"] for r in rows],
                         ["19.8", "19.8", "19.7", "20.51"])
        self.assertFalse(reviews)

    def test_complex_reference_is_flagged(self):
        headers = stage2.reference_headers([
            word("reference", 54, 88, 80), word("Haramura", 364, 400, 80),
            word("in", 403, 409, 80), word("Compston70", 415, 461, 80),
            word("Wakita70", 466, 500, 80),
        ], 80, 128)
        self.assertEqual(headers[0]["text"], "Haramura in Compston70")
        self.assertTrue(headers[0]["complex"])

    def test_blank_feo_row_is_not_zero(self):
        document = {"source": {"document_id": "doc", "atlas_sample_ids": ["x"]}}
        page = {
            "page_number": 1,
            "word_coordinates": [word("reference", 54, 88, 80), word("FeO", 54, 70, 128)],
            "chemistry_layout": {
                "reference_rows": [{"y0": 80, "text": "reference"}],
                "feo_rows": [{"y0": 128, "text": "FeO"}],
                "technique_rows": [],
            },
        }
        rows, _ = stage2.extract_feo_from_page(document, page)
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
