import tempfile
import unittest
from pathlib import Path

from build_relab_atlas_crosswalk import build_crosswalk, parse_relab_xml


ATLAS = [
    {
        "atlas_sample_id": "76535", "sample_type": "rock", "lithology": "troctolite",
        "description": "Coarse-grained sample.", "detail_url": "", "atlas_pdf_url": "",
    },
    {
        "atlas_sample_id": "10084", "sample_type": "soil", "lithology": "regolith",
        "description": "Bulk lunar soil.", "detail_url": "", "atlas_pdf_url": "",
    },
]


XML = """<?xml version="1.0"?>
<Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1"
 xmlns:speclib="http://pds.nasa.gov/pds4/speclib/v1">
 <Identification_Area>
  <logical_identifier>urn:nasa:pds:relab:data_reflectance:c1ls04</logical_identifier>
  <version_id>1.1</version_id><title>RELAB Spectrum ID c1ls04</title>
 </Identification_Area>
 <Observation_Area><Discipline_Area>
  <speclib:Specimen_Parameters>
   <speclib:specimen_id>LS-CMP-004</speclib:specimen_id>
   <speclib:specimen_name>76535,93</speclib:specimen_name>
   <speclib:specimen_description>Plagioclase mineral separate</speclib:specimen_description>
   <speclib:specimen_collection_location>Apollo 17</speclib:specimen_collection_location>
  </speclib:Specimen_Parameters>
  <speclib:Specimen_Classification>
   <speclib:specimen_type>Returned Lunar Sample</speclib:specimen_type>
   <speclib:material_origin>Natural</speclib:material_origin>
   <speclib:material_type>Mineral</speclib:material_type>
   <speclib:material_subtype>Particulate</speclib:material_subtype>
  </speclib:Specimen_Classification>
  <speclib:Reflectance_Parameters>
   <speclib:spectral_range_min>350</speclib:spectral_range_min>
   <speclib:spectral_range_max>2600</speclib:spectral_range_max>
   <speclib:spectral_range_unit_name>nm</speclib:spectral_range_unit_name>
   <speclib:incidence_angle unit="deg">30</speclib:incidence_angle>
   <speclib:emission_angle unit="deg">0</speclib:emission_angle>
   <speclib:phase_angle unit="deg">30</speclib:phase_angle>
  </speclib:Reflectance_Parameters>
 </Discipline_Area></Observation_Area>
 <File_Area_Observational><File><file_name>c1ls04.tab</file_name></File></File_Area_Observational>
</Product_Observational>
"""


class CrosswalkTests(unittest.TestCase):
    def test_xml_parsing_and_subsample_match(self):
        atlas = ATLAS
        self.assertEqual(len(atlas), 2)
        with tempfile.TemporaryDirectory() as directory:
            xml_path = Path(directory) / "c1ls04.xml"
            xml_path.write_text(XML, encoding="utf-8")
            measurement = parse_relab_xml(xml_path)
            self.assertIsNotNone(measurement)
            self.assertEqual(measurement["measurement_id"], "c1ls04")
            self.assertEqual(measurement["measurement_data_file"], "c1ls04.tab")
            row = build_crosswalk([measurement], atlas)[0]
            self.assertEqual(row["match_type"], "subsample_or_fraction")
            self.assertEqual(row["apollo_parent_sample"], "76535")
            self.assertEqual(row["needs_review"], "yes")

    def test_literal_sample_number_match(self):
        atlas = ATLAS
        measurement = {
            "measurement_id": "example",
            "specimen_id": "LS-TEST-001",
            "specimen_name": "10084",
        }
        row = build_crosswalk([measurement], atlas)[0]
        self.assertEqual(row["match_type"], "sample_number")
        self.assertEqual(row["needs_review"], "no")

    def test_described_fraction_is_not_exact(self):
        atlas = [{
            "atlas_sample_id": "79221", "sample_type": "soil", "lithology": "",
            "description": "", "detail_url": "", "atlas_pdf_url": "",
        }]
        measurement = {
            "measurement_id": "fraction",
            "specimen_id": "LS-BWD-149",
            "specimen_name": "79221 bulk 125-250 um",
        }
        row = build_crosswalk([measurement], atlas)[0]
        self.assertEqual(row["match_type"], "subsample_or_fraction")
        self.assertEqual(row["apollo_parent_sample"], "79221")
        self.assertEqual(row["needs_review"], "yes")

    def test_dot_subsample(self):
        atlas = [{
            "atlas_sample_id": "15531", "sample_type": "rock", "lithology": "",
            "description": "", "detail_url": "", "atlas_pdf_url": "",
        }]
        measurement = {
            "measurement_id": "dot",
            "specimen_id": "LS-GJT-126",
            "specimen_name": "15531.22 <150 um",
        }
        row = build_crosswalk([measurement], atlas)[0]
        self.assertEqual(row["normalized_specimen_name"], "15531,22")
        self.assertEqual(row["match_type"], "subsample_or_fraction")

    def test_grouped_parent_and_subsample(self):
        atlas = [{
            "atlas_sample_id": "10003", "sample_type": "soil", "lithology": "",
            "description": "", "detail_url": "", "atlas_pdf_url": "",
        }]
        measurement = {
            "measurement_id": "grouped",
            "specimen_id": "LS-CMP-021",
            "specimen_name": "10,003,173",
        }
        row = build_crosswalk([measurement], atlas)[0]
        self.assertEqual(row["apollo_parent_sample"], "10003")
        self.assertEqual(row["normalized_specimen_name"], "10003,173")
        self.assertEqual(row["match_type"], "subsample_or_fraction")

    def test_luna_number_is_not_apollo(self):
        atlas = [{
            "atlas_sample_id": "24160", "sample_type": "", "lithology": "",
            "description": "", "detail_url": "", "atlas_pdf_url": "",
        }]
        measurement = {
            "measurement_id": "luna",
            "specimen_id": "LR-CMP-135",
            "specimen_name": "Luna 24, 24160",
            "specimen_collection_location": "Luna 24",
        }
        row = build_crosswalk([measurement], atlas)[0]
        self.assertEqual(row["match_type"], "non_apollo")
        self.assertEqual(row["apollo_parent_sample"], "")
        self.assertEqual(row["needs_review"], "no")

    def test_recognizable_apollo_sample_missing_from_index(self):
        measurement = {
            "measurement_id": "missing",
            "specimen_id": "LR-DSM-230",
            "specimen_name": "14003,96 Respirable",
        }
        row = build_crosswalk([measurement], [])[0]
        self.assertEqual(row["match_type"], "atlas_missing")
        self.assertEqual(row["apollo_parent_sample"], "14003")
        self.assertEqual(row["needs_review"], "yes")


if __name__ == "__main__":
    unittest.main()
