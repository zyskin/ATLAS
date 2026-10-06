#!/usr/bin/env python3
"""Small synthetic checks; no project data are needed."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


class PreliminaryMLTest(unittest.TestCase):
    def test_builder_preserves_stable_ids_and_averages(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            comprehensive = root / "comprehensive.csv"
            columns = [
                "id", "i_atlas_sample", "j_child", "k_feo_wt_percent", "feo_known",
                "relab_measurement_id", "relab_specimen_id", "relab_specimen_name",
                "chemistry_product_id", "chemistry_specimen_id", "chemistry_phase",
                "chemistry_source_and_method", "atlas_sample_type", "atlas_lithology",
                "crosswalk_needs_review",
            ]
            rows = [
                dict(zip(columns, ["44", "10003", "1", "10", "yes", "m1", "s1",
                                   "sample", "p1", "s1", "Bulk", "source A", "soil",
                                   "", "no"])),
                dict(zip(columns, ["41", "10003", "1", "12", "yes", "m1", "s1",
                                   "sample", "p2", "s1", "Bulk", "source B", "soil",
                                   "", "no"])),
            ]
            with comprehensive.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader(); writer.writerows(rows)

            spectra = root / "spectra.csv"
            with spectra.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "measurement_id", "m3_band", "m3_response_centroid_nm",
                    "m3_reflectance", "usable",
                ])
                writer.writeheader()
                for band in range(1, 75):
                    writer.writerow({"measurement_id": "m1", "m3_band": band,
                                     "m3_response_centroid_nm": 400 + band * 20,
                                     "m3_reflectance": 0.1 + band / 1000,
                                     "usable": "yes"})

            output = root / "dataset.csv"
            manifest = root / "manifest.json"
            subprocess.run([
                sys.executable, str(ROOT / "build_m3_feo_ml_dataset.py"),
                "--comprehensive-table", str(comprehensive),
                "--m3-spectra", str(spectra),
                "--output-csv", str(output),
                "--manifest-json", str(manifest),
            ], check=True, capture_output=True, text=True)
            with output.open(newline="", encoding="utf-8") as handle:
                result = list(csv.DictReader(handle))
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["id"], "41")
            self.assertEqual(result[0]["comprehensive_table_ids"], "41;44")
            self.assertAlmostEqual(float(result[0]["feo_mean_wt_percent"]), 11.0)
            self.assertEqual(json.loads(manifest.read_text())["row_count"], 1)


if __name__ == "__main__":
    unittest.main()
