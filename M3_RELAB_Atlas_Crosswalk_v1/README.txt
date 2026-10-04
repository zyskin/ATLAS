LPI LUNAR SAMPLE ATLAS / RELAB CROSSWALK -- VERSION 1
=====================================================

PURPOSE
-------
This code creates:

1. atlas_samples.csv
   One row per sample in the LPI Lunar Sample Atlas index.

2. relab_measurements.csv
   One row per downloaded RELAB reflectance measurement label.

3. relab_atlas_crosswalk.csv
   A documented link from each RELAB measurement/specimen to an Atlas sample.

The reflectance values themselves are NOT needed for this matching stage.
The program uses RELAB specimen metadata, especially specimen_name.


MINIMAL ENVIRONMENT
-------------------
Python 3.10 or newer is recommended.

Create a small conda environment:

    conda create -n relab-atlas python=3.12 -c conda-forge
    conda activate relab-atlas

No GDAL, NumPy, local language model, GPU, or PDF software is required for
this catalogue/crosswalk stage.


STEP 1 -- COLLECT THE ATLAS INDEX
---------------------------------
Try the live website first:

    python collect_atlas_index.py --output atlas_samples.csv

The LPI website may return an HTTP 403 anti-bot page to command-line tools.
If that happens:

1. Open this page in Firefox or Chrome:
   https://www.lpi.usra.edu/lunar/samples/atlas/index.cfm

2. Use File -> Save Page As, and save the HTML page, for example as:
   lunar_sample_atlas.html

3. Run:

    python collect_atlas_index.py \
      --input-html lunar_sample_atlas.html \
      --output atlas_samples.csv

Outputs:

    atlas_samples.csv
    atlas_samples.metadata.json

The metadata JSON records the source, date, record count, and SHA-256 hash of
the source HTML. This makes later runs reproducible.


STEP 2 -- PREPARE RELAB METADATA AND BUILD THE CROSSWALK
---------------------------------------------------------
If the downloaded RELAB material contains PDS4 XML measurement labels:

    python build_relab_atlas_crosswalk.py \
      --atlas-csv atlas_samples.csv \
      --relab-dir /path/to/downloaded/relab \
      --output-dir crosswalk_output

The directory is searched recursively. Chemistry XML, collection inventories,
and other unrelated XML files are ignored. Only RELAB reflectance measurement
labels containing specimen and wavelength-range metadata are used.

Alternatively, if RELAB metadata has already been exported as CSV or TSV:

    python build_relab_atlas_crosswalk.py \
      --atlas-csv atlas_samples.csv \
      --relab-csv relab_export.csv \
      --output-dir crosswalk_output

The CSV must contain a specimen-name or sample-number column. Common headings
such as specimen_name, Specimen Name, sample_id, and sample_number are
recognized automatically.

Outputs:

    crosswalk_output/relab_measurements.csv
    crosswalk_output/relab_atlas_crosswalk.csv
    crosswalk_output/crosswalk_summary.json


MEANING OF MATCH TYPES
----------------------
exact
    The RELAB specimen name is exactly the same as an Atlas sample number.

parent
    RELAB contains a subsample such as 76535,93, while the Atlas index contains
    parent sample 76535. This is a useful link, but does not prove that every
    composition reported for 76535 applies exactly to subsample 76535,93.

ambiguous
    More than one Atlas row could match. Human review is required.

unmatched
    No reliable five-digit Apollo parent number was found, or the number was
    absent from the Atlas index.

The needs_review column is yes for parent, ambiguous, and unmatched records.


WHAT THIS VERSION DELIBERATELY DOES NOT DO
------------------------------------------
- It does not read reflectance values from .tab files.
- It does not download or interpret Atlas PDFs.
- It does not extract FeO or modal mineralogy.
- It does not use a language model.

Those are later stages. PDF extraction should use ordinary text/table tools
first. A pretrained local language model can then interpret only the difficult
narrative passages and return a constrained JSON record with page number and
supporting text. No model training is required for the first implementation.


VALIDATION
----------
Run the supplied self-test from this directory:

    python -m unittest -v test_crosswalk.py

The test checks an exact Atlas match and a parent/subsample match using a small
representative PDS4 RELAB XML label.
