# Adams–RELAB parent overlay plots

This directory makes one plot for every validated Adams spectrum whose Apollo
parent also has one or more RELAB spectral measurements.

For each Adams spectrum, the plot contains:

- every RELAB measurement attached to the same Apollo parent, including a
  direct parent record, child/subsample, grain-size fraction, separate, or other
  prepared material;
- all RELAB curves in the same blue color;
- one Adams curve in red;
- a protected top-left annotation containing the Adams specimen/measurement IDs,
  ATLAS parent ID, and FeO value;
- by default, the Adams reflectance multiplied by the wavelength-dependent
  **parent-balanced mean RELAB/Adams coefficient**.

The program does not claim that an Adams specimen and a RELAB child are the
same physical subsample. Their supported relationship is normally only that
both are linked to the same Apollo parent.

## Inputs

The runner uses these project files:

```text
data/adams_id_lists/adams_atlas_crosswalk_validated.csv
data/adams-20220223T225406Z-001/adams/wavelength_corrected_data/
crosswalk_output_v2_2026-10-03/relab_atlas_crosswalk.csv
relab_raw_native_2026-10-04/relab_spectrum_index.csv
relab_raw_native_2026-10-04/relab_spectral_points.csv
relab_raw_native_2026-10-04/atlas_relab_feo_table.csv
data/structured_v1/atlas_feo_measurements.csv
data/ai_resolution_v1/review_candidate_adjudications.csv
data/ai_resolution_v1/missing_case_resolutions.csv
m3_feo_ml_preliminary/manual_feo_claims_reviewed.csv
data/relab_adams_ratio_v1/relab_adams_ratio_summary_short_parent.csv
Calibration/M3G20081211_RDN_BPF.csv
```

The RELAB points table is large, so it is scanned once for all required
measurements rather than once per figure.

## FeO annotation and provenance

The annotation is always three rows. If no defensible FeO value can be found,
the third row is deliberately left as `FeO:` with no number.

The Bhatt-derived `literature_feo_wt_percent` column in the Adams crosswalk is
intentionally ignored. FeO is selected only from the confirmed recovery
pipeline, in this order:

1. Exact chemistry for RELAB measurements actually included in that plot is
   read from
   `relab_raw_native_2026-10-04/atlas_relab_feo_table.csv`. Only rows with
   `feo_known` true and a nonnegative `k_feo_wt_percent` are accepted.
2. If that does not exist, accepted/reviewed parent or documented-fraction
   claims are sought in:

   - `data/structured_v1/atlas_feo_measurements.csv`: only
     `needs_review=no` and `sample_scope=bulk_or_whole_sample`;
   - `data/ai_resolution_v1/review_candidate_adjudications.csv`: only
     `decision=accepted_parent_or_documented_fraction`;
   - `data/ai_resolution_v1/missing_case_resolutions.csv`: only statuses
     beginning with `resolved_`;
   - `m3_feo_ml_preliminary/manual_feo_claims_reviewed.csv`: only
     `include=yes`.

When a tier has several distinct accepted values, the plot displays
their arithmetic mean, range, and count. The plot manifest preserves the
source tier, confidence tier, range, count, and provenance details. Unreviewed
PDF candidates and mineral/component-only values are not used. The program
also does not transfer a value merely because parent and child sample numbers
resemble one another. The bracketed source label on the plot distinguishes a
mean of confirmed RELAB chemistry from an accepted literature value; neither
is presented as a direct Adams-subsample measurement.

## Run all plots

From the `M3_RELAB_Atlas_Crosswalk_v1` directory:

```bash
python adams_relab_parent_overlays/run_all_parent_overlays.py \
  --project-root "$PWD"
```

With the currently committed crosswalks, the default selection should produce
39 PNG files: one for each Adams spectrum sharing a known Apollo parent number
with at least one RELAB record. This includes parent `14003`, whose Apollo
parent number is known on both sides but whose page is absent from the local
Atlas index.

To require an actual matching page in the local Atlas index, run:

```bash
python adams_relab_parent_overlays/run_all_parent_overlays.py \
  --project-root "$PWD" \
  --require-atlas-page
```

That stricter selection should currently produce 38 PNG files.

## Output

The default output directory is:

```text
adams_relab_parent_overlays/plots/
```

It contains:

- one PNG per Adams spectrum;
- `plot_manifest.csv`, one row per PNG, with the parent, Adams identifiers,
  FeO value and provenance, relationship type, counts, and complete lists of
  included RELAB measurement and specimen IDs;
- `plot_relab_records.csv`, one row for every plot–RELAB combination, including
  the RELAB specimen name, parent/child relationship, instrument, geometry, and
  wavelength coverage.

Existing PNG files are redrawn by default. Use `--skip-existing` to retain
them while rebuilding the manifests.

## Coefficient choices

The default is:

```text
--coefficient parent
```

This uses the parent-balanced coefficient because each figure is organized by
Apollo parent. The historical file containing `child` in its name is not a
physical child-only coefficient; it is a pair-weighted coefficient. It can be
selected explicitly with:

```text
--coefficient child
```

For an unscaled Adams curve, use:

```text
--coefficient none
```

The coefficient is linearly interpolated between the response-weighted M3 band
centres. The nearest endpoint coefficient is held constant outside that range.

## Pull the code and later push generated plots

```bash
cd /media/maxim/Elements/grants/2024/INSRU/pgdal/M3_RELAB_Atlas_Crosswalk_v1
git pull --rebase origin main

python adams_relab_parent_overlays/run_all_parent_overlays.py \
  --project-root "$PWD"

git add adams_relab_parent_overlays/plots
git commit -m "Add Adams RELAB parent overlay plots"
git push origin main
```

Before committing, `git status --short` can be used to review the generated
PNG files and manifests.
