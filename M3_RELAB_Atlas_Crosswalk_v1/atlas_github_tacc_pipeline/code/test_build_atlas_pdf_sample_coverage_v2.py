#!/usr/bin/env python3

import csv
import gzip
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROGRAM = Path(__file__).with_name("build_atlas_pdf_sample_coverage_v2.py")


class CoverageTests(unittest.TestCase):
    def test_indirect_split_is_found_but_not_accepted_as_chemistry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atlas = root / "atlas.csv"
            with atlas.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["atlas_sample_id", "mission", "detail_url"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"atlas_sample_id": "15030", "mission": "Apollo 15", "detail_url": "a"},
                        {"atlas_sample_id": "15040", "mission": "Apollo 15", "detail_url": "b"},
                        {"atlas_sample_id": "15041", "mission": "Apollo 15", "detail_url": "c"},
                        {"atlas_sample_id": "15042", "mission": "Apollo 15", "detail_url": "d"},
                        {"atlas_sample_id": "10072A", "mission": "Apollo 11", "detail_url": "e"},
                    ]
                )
            parsed = root / "parsed" / "json_gz"
            parsed.mkdir(parents=True)
            document = {
                "source": {
                    "document_id": "15030__compendium__test",
                    "atlas_sample_ids": ["15030", "15040"],
                    "source_url": "https://example.test/compendium/15030.pdf",
                },
                "pages": [
                    {
                        "page_number": 1,
                        "text": "15030 and 15040\nTrench Soils",
                        "chemistry_layout": {"feo_rows": []},
                    },
                    {
                        "page_number": 8,
                        "text": "Table 2. Chemical composition of 15040.\nFeO 14.5 14.3",
                        "chemistry_layout": {"feo_rows": [{"text": "FeO 14.5 14.3"}]},
                    },
                    {
                        "page_number": 9,
                        "text": (
                            "Processing and allocation\n15040 parent soil\n"
                            "15041 split for analysis\n10072A is separately indexed"
                        ),
                        "chemistry_layout": {"feo_rows": []},
                    },
                ],
            }
            with gzip.open(parsed / "doc.json.gz", "wt", encoding="utf-8") as handle:
                json.dump(document, handle)
            output = root / "out"
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROGRAM),
                    "--atlas-csv", str(atlas),
                    "--parsed-dir", str(root / "parsed"),
                    "--output-dir", str(output),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with (output / "atlas_pdf_sample_coverage_combined.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            row_15041 = next(row for row in rows if row["atlas_sample_id"] == "15041")
            self.assertEqual(row_15041["association_source"], "pdf_text_indirect")
            self.assertEqual(row_15041["relation_type"], "allocation_or_split")
            self.assertEqual(row_15041["association_needs_review"], "yes")
            self.assertEqual(row_15041["document_has_feo_rows"], "yes")
            self.assertEqual(row_15041["document_feo_pages"], "8")
            row_10072a = next(row for row in rows if row["atlas_sample_id"] == "10072A")
            self.assertEqual(row_10072a["association_source"], "pdf_text_indirect")
            with (output / "atlas_samples_without_pdf_coverage.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                uncovered = list(csv.DictReader(handle))
            self.assertEqual([row["atlas_sample_id"] for row in uncovered], ["15042"])


if __name__ == "__main__":
    unittest.main()
