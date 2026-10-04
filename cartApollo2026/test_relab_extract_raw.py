import csv
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from relab_extract_raw import (
    chemistry_label_info,
    extract_chemistry,
    extract_spectrum,
    reflectance_label_info,
)


ROOT = Path(__file__).resolve().parent
FIXTURE_ROOT = Path(os.environ.get("RELAB_TEST_DIR", ROOT / "upload"))


def fixture(name: str) -> Path:
    matches = list(FIXTURE_ROOT.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {name!r} below {FIXTURE_ROOT}, found {len(matches)}. "
            "Set RELAB_TEST_DIR to the RELAB download root."
        )
    return matches[0]


class RelabRawExtractorTests(unittest.TestCase):
    def test_three_column_native_grid_and_footer_exclusion(self):
        info = reflectance_label_info(fixture("c1lr154.xml"))
        self.assertIsNotNone(info)
        index, points, warnings = extract_spectrum(info, hash_inputs=False)
        self.assertEqual(index["measurement_id"], "c1lr154")
        self.assertEqual(index["native_point_count"], "465")
        self.assertEqual(index["native_field_count"], "3")
        self.assertEqual(index["standard_deviation_present"], "yes")
        self.assertEqual(len(points), 465)
        self.assertEqual(points[0]["wavelength_nm"], "280")
        self.assertEqual(points[-1]["wavelength_nm"], "2600")
        self.assertEqual(points[0]["wavelength_original"], "280.0")
        self.assertEqual(points[0]["reflectance"], "0.05149")
        self.assertEqual(points[0]["uncertainty_value"], "0.02507")
        self.assertEqual(points[0]["uncertainty_type"], "Standard_Deviation")
        self.assertEqual(warnings, [])

    def test_two_column_native_grid(self):
        info = reflectance_label_info(fixture("c1gl15.xml"))
        self.assertIsNotNone(info)
        index, points, warnings = extract_spectrum(info, hash_inputs=False)
        self.assertEqual(index["measurement_id"], "c1gl15")
        self.assertEqual(index["native_point_count"], "226")
        self.assertEqual(index["native_field_count"], "2")
        self.assertEqual(index["standard_deviation_present"], "no")
        self.assertEqual(index["wavelength_median_step_nm"], "10")
        self.assertEqual(len(points), 226)
        self.assertEqual(points[0]["wavelength_nm"], "350")
        self.assertEqual(points[-1]["wavelength_nm"], "2600")
        self.assertEqual(points[0]["uncertainty_value"], "")
        self.assertEqual(warnings, [])

    def test_chemistry_feo_is_read_by_label_position(self):
        info = chemistry_label_info(fixture("lr-cmp-154_specimenchemistry.xml"))
        self.assertIsNotNone(info)
        rows, columns = extract_chemistry(info, hash_inputs=False)
        self.assertEqual(len(rows), 1)
        self.assertIn("feo_percent", columns)
        self.assertEqual(rows[0]["specimen_id"], "LR-CMP-154")
        self.assertEqual(rows[0]["phase_name"], "Bulk")
        self.assertEqual(rows[0]["feo_percent"], "22.47")
        self.assertEqual(rows[0]["sources_and_methods"], "PGI, Taylor et al.")

    def test_cli_bounded_extraction(self):
        from relab_extract_raw import main

        with tempfile.TemporaryDirectory() as directory:
            fixture_dir = Path(directory) / "input"
            fixture_dir.mkdir()
            for name in (
                "c1lr154.xml",
                "c1lr154.tab",
                "c1gl15.xml",
                "c1gl15.tab",
                "lr-cmp-154_specimenchemistry.xml",
                "lr-cmp-154_specimenchemistry.csv",
            ):
                shutil.copy2(fixture(name), fixture_dir / name)
            output_dir = Path(directory) / "output"
            status = main(
                [
                    "--relab-dir",
                    str(fixture_dir),
                    "--output-dir",
                    str(output_dir),
                    "--measurement-ids",
                    "c1lr154,c1gl15",
                    "--hash-inputs",
                ]
            )
            self.assertEqual(status, 0)
            with (output_dir / "relab_spectrum_index.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                index_rows = list(csv.DictReader(handle))
            self.assertEqual({row["measurement_id"] for row in index_rows}, {"c1lr154", "c1gl15"})
            with (output_dir / "relab_chemistry.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                chemistry_rows = list(csv.DictReader(handle))
            self.assertEqual(chemistry_rows[0]["feo_percent"], "22.47")


if __name__ == "__main__":
    unittest.main()
