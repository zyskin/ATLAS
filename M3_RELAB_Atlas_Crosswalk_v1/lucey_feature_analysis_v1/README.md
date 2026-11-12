# Lucey-plane plots and preliminary RELAB spectral classification

This directory contains reproducible Python and Wolfram Language programs for
four separate tasks:

1. place RELAB spectra with known FeO in the three-dimensional coordinates
   `(R750, R950/R750, FeO)`;
2. place every valid pixel of an extracted M3 snapshot in the two-dimensional
   Lucey plane;
3. calculate spectral-shape features `S`, `D1`, `D2`, and mean reflectivity,
   apply explicitly editable preliminary mineral-like rules, and compare them
   with words found in RELAB/Atlas metadata;
4. make one ground-truth-versus-held-out-prediction plot for every ML,
   Lucey, and Kumar method in a cross-validation prediction table.

The coordinate order used everywhere is:

```text
x = R750
y = R950 / R750
```

The marked origins are:

```text
Lucey: (0.08, 1.19)
Kumar: (0.08, 1.18)
```

## Existing source files

The prepared RELAB spectra convolved with the official M3 global-mode band
responses are in:

```text
M3_RELAB_Atlas_Crosswalk_v1/data/relab_m3_v1/relab_m3_spectra.csv
```

It is a long table: one row per RELAB measurement and M3 band.  Its columns
include `measurement_id`, `m3_band`, `m3_response_centroid_nm`,
`m3_reflectance`, `bandpass_coverage_fraction`, and `usable`.

The code that produced it is:

```text
M3_RELAB_Atlas_Crosswalk_v1/relab_to_m3_bandpasses.py
```

That program reconstructs each native RELAB curve with PCHIP and integrates it
against the official M3 bandpass response.  This is not the natural-cubic-
spline code.

Natural cubic splines were used in:

```text
M3_RELAB_Atlas_Crosswalk_v1/compare_relab_adams_m3_ratios_v4.py
```

The relevant line is:

```python
CubicSpline(x, y, bc_type="natural", extrapolate=False)
```

On the Ubuntu machine, locate every such use with:

```bash
PGDAL=/media/maxim/Elements/grants/2024/INSRU/pgdal
rg -n 'CubicSpline|bc_type="natural"' "$PGDAL/M3_RELAB_Atlas_Crosswalk_v1"
```

## Install the Python requirements

The existing `qwen-local` environment is adequate after installing any missing
packages:

```bash
conda activate qwen-local
ATLAS=/media/maxim/Elements/grants/2024/INSRU/pgdal/M3_RELAB_Atlas_Crosswalk_v1
python -m pip install -r "$ATLAS/lucey_feature_analysis_v1/requirements.txt"
```

## 1. RELAB spectra in three dimensions

Python:

```bash
ATLAS=/media/maxim/Elements/grants/2024/INSRU/pgdal/M3_RELAB_Atlas_Crosswalk_v1
CODE="$ATLAS/lucey_feature_analysis_v1"
OUT="$ATLAS/data/lucey_feature_analysis_v1"
mkdir -p "$OUT"

python "$CODE/plot_relab_lucey_3d.py" \
  --ground-truth "$ATLAS/data/ml_combined_v2/m3_feo_isfeo_ground_truth_v2.csv" \
  --target combined \
  --output-png "$OUT/relab_lucey_3d_combined.png"
```

For only directly associated specimen-level FeO, replace `combined` with
`direct`.  The program also writes the exact plotted records to a CSV and a
JSON summary beside the PNG.

Mathematica/Wolfram Language:

```text
PlotRELABLucey3D.wl
```

Edit the configuration block at its beginning, then evaluate the file.  Change
`targetColumn` to `feo_direct_wt_percent_mean` for the direct-only case.

## 2. All M3 pixels in the two-dimensional Lucey plane

The default hexagonal-density rendering uses every valid pixel.  It aggregates
pixels spatially only on the plot canvas; it does not randomly subsample them.

```bash
OLD=/media/maxim/Elements/grants/2024/INSRU/pgdal/m3_stage1_retest_2026-10-01/all_memmap
OUT=/media/maxim/Elements/grants/2024/INSRU/pgdal/m3_lucey_plots

python "$CODE/plot_m3_pixels_lucey_2d.py" \
  --stage1-dir "$OLD" \
  --prefix M3G20090107T011405 \
  --definition both \
  --render hexbin \
  --output-dir "$OUT"
```

`direct` uses band 9 for 750 nm and band 19 for 950 nm.  `kumar_mean` uses the
means of bands 8--10 and 18--20.  `both` produces both plots.  Nodata,
non-finite, nonpositive-750-nm, and explicitly out-of-range values are excluded
and counted in the output manifest.  The plotting limits are command-line
parameters rather than silently hard-coded clipping.

The Mathematica equivalent is:

```text
PlotM3PixelsLucey2D.wl
```

Its configuration block already points to the all-band memmap extraction used
in the recent test.

## 3. S, D1, D2, mean reflectivity, and preliminary labels

Definitions used by the supplied default rules:

- `S` is the reflectance slope from 750 to 1500 nm, divided by `R750`, in
  inverse micrometres;
- `D1` is the maximum continuum-removed band depth from 850 to 1250 nm, with a
  straight continuum between 750 and 1500 nm;
- `D2` is the maximum continuum-removed band depth from 1800 to 2300 nm, with a
  straight continuum between 1500 and 2500 nm;
- mean reflectivity is the wavelength average from 750 to 2500 nm.

A natural cubic spline through the M3-response-centroid points is evaluated on
a 2 nm grid.  No extrapolation is allowed. A spectrum that does not cover the
full 750--2500 nm diagnostic interval remains in the output: every available
feature is recorded, its missing features are named, and no mineral-like class
is asserted from incomplete D1/D2 information. The rules in `mineral_rules.json`
are intentionally editable and yield labels such as `pyroxene_like` and
`olivine_like`, not asserted mineral abundances.

```bash
python "$CODE/relab_spectral_features_and_mineralogy.py" \
  --m3-spectra "$ATLAS/data/relab_m3_v1/relab_m3_spectra.csv" \
  --rules-json "$CODE/mineral_rules.json" \
  --spectrum-index "$ATLAS/relab_raw_native_2026-10-04/relab_spectrum_index.csv" \
  --comprehensive-table "$ATLAS/relab_raw_native_2026-10-04/atlas_relab_feo_table.csv" \
  --ground-truth "$ATLAS/data/ml_combined_v2/m3_feo_isfeo_ground_truth_v2.csv" \
  --atlas-enriched "$ATLAS/data/structured_v1/atlas_samples_enriched.csv" \
  --output-csv "$OUT/relab_spectral_features_mineralogy.csv"
```

The CSV records the spectral rule, the metadata keyword labels, the source text
used for those metadata labels, and whether the two classifications agree.
Disagreement is a review flag, not proof that either side is wrong; many lunar
samples are mixtures.

## 4. FeO ground truth versus every held-out prediction

Use cross-validation predictions, not predictions obtained by fitting and then
predicting the same training rows.

```bash
PRED="$ATLAS/data/ml_combined_v2/results_combined_quick/cv_predictions_by_measurement.csv"

python "$CODE/plot_feo_prediction_scatter.py" \
  --predictions "$PRED" \
  --target-column feo_wt_percent_mean \
  --unit apollo_parent_mean \
  --output-dir "$OUT/feo_prediction_scatter_combined"
```

The program automatically detects every `prediction_*` column, including all
ML models and all Lucey/Kumar variants, and writes one PNG per method plus a
metric summary CSV.  For the direct-only experiment use its corresponding
prediction table and target `feo_direct_wt_percent_mean`.

