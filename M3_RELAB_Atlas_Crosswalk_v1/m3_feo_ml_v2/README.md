# Combined FeO / Is-FeO ground truth and ML v2

This directory rebuilds the laboratory ground truth after the Atlas PDF review,
publication review, JSC lunar-soil extraction, and regolith-breccia extraction.
It does not replace the source tables.  Every accepted value remains traceable
to its source in the long claims table.

## Current coverage

The 2026-11-02 build produced:

- 524 RELAB measurements with 74 usable M3-forward-modelled bands;
- 108 Apollo parent groups;
- 523 measurements / 107 parents with FeO;
- 113 measurements / 29 parents with Is/FeO;
- 456 unique source claims represented in the joined data: 427 FeO and 29 Is/FeO.

FeO association quality is not uniform:

- 66 rows use only chemistry explicitly associated with the RELAB measurement;
- 39 combine exact and Apollo-parent claims;
- 418 use Apollo-parent claims provisionally attached to a child/fraction spectrum;
- 1 row has Is/FeO but no accepted FeO.

The parent grouping is mandatory in cross-validation.  A spectrum from the same
Apollo parent never appears in both training and test sets.

## Output files

`data/ml_combined_v2/m3_feo_isfeo_ground_truth_v2.csv`
: One row per RELAB measurement.  Contains stable old IDs, parent/child identity,
  all accepted FeO and Is/FeO values, source-resolved JSON, eight Lucey/Kumar
  baseline variants, and M3 bands 1--74.

`data/ml_combined_v2/m3_feo_isfeo_claims_v2.csv`
: One row per unique source claim.  This is the easiest file for auditing where
  a value came from and which spectra inherited it.

`data/ml_combined_v2/m3_feo_isfeo_parent_summary_v2.csv`
: One row per Apollo parent, with linked spectra and separate direct, parent,
  combined FeO, and Is/FeO summaries.

`data/ml_combined_v2/m3_feo_isfeo_ground_truth_v2_manifest.json`
: Input hashes, counts, exclusions, constants, and warnings.

`data/ml_combined_v2/results_combined_quick/`
: Quick grouped-CV run using the combined accepted FeO mean.

`data/ml_combined_v2/results_direct_only_quick/`
: Sensitivity run using only FeO explicitly associated with the RELAB
  measurement.  It has 105 spectra from 33 Apollo parents.

The scripts also generate fitted `.joblib` models locally.  Those binary
deployment fits are intentionally not committed because the scientifically
relevant evidence is the held-out CV predictions, metrics, parameters, and
plots; the fits can be regenerated in seconds.

## Baselines included

The run intentionally keeps both angle conventions:

- principal angle: `-atan((y-y0)/(x-x0))`;
- quadrant-aware angle: `(-atan2(y-y0,x-x0)) mod 2*pi`.

It reports the standard Lucey origin `(0.08, 1.19)`, the previously saved
project origin `(0.03433418510481713, 1.2571662682003644)`, and the Kumar
three-band averages around 750 and 950 nm.  The origin-changed coefficients are
project-fit constants, not presented as a published universal calibration.

## Rebuild

From `M3_RELAB_Atlas_Crosswalk_v1`:

```bash
python m3_feo_ml_v2/build_combined_ground_truth_v2.py \
  --comprehensive-table relab_raw_native_2026-10-04/atlas_relab_feo_table.csv \
  --m3-spectra data/relab_m3_v1/relab_m3_spectra.csv \
  --atlas-pdf-feo data/structured_v1/atlas_feo_measurements.csv \
  --adjudications data/ai_resolution_v1/review_candidate_adjudications.csv \
  --missing-resolutions data/ai_resolution_v1/missing_case_resolutions.csv \
  --manual-feo m3_feo_ml_preliminary/manual_feo_claims_reviewed.csv \
  --source-chemistry-index data/source_extracts/source_chemistry_index.csv \
  --existing-is-feo data/is_feo_v1/existing_is_feo_records.csv \
  --output-csv data/ml_combined_v2/m3_feo_isfeo_ground_truth_v2.csv \
  --manifest-json data/ml_combined_v2/m3_feo_isfeo_ground_truth_v2_manifest.json
```

Combined provisional run:

```bash
python m3_feo_ml_v2/train_combined_feo_models_v2.py \
  --dataset data/ml_combined_v2/m3_feo_isfeo_ground_truth_v2.csv \
  --output-dir data/ml_combined_v2/results_combined_quick \
  --target-column feo_wt_percent_mean \
  --search-level quick
```

Exact-association sensitivity run:

```bash
python m3_feo_ml_v2/train_combined_feo_models_v2.py \
  --dataset data/ml_combined_v2/m3_feo_isfeo_ground_truth_v2.csv \
  --output-dir data/ml_combined_v2/results_direct_only_quick \
  --target-column feo_direct_wt_percent_mean \
  --search-level quick
```

The primary comparison is the `apollo_parent_mean` section of `cv_metrics.csv`.
Measurement-level scores are secondary because some parents have many more
spectra than others.
