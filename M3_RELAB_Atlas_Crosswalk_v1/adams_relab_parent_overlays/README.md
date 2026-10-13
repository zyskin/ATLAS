# Adams–RELAB parent overlay plots

This directory makes one plot for every validated Adams spectrum whose Apollo
parent also has one or more RELAB spectral measurements.

For each Adams spectrum, the plot contains:

- every RELAB measurement attached to the same Apollo parent, including a
  direct parent record, child/subsample, grain-size fraction, separate, or other
  prepared material;
- all RELAB curves in the same blue color;
- one Adams curve in red;
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
data/relab_adams_ratio_v1/relab_adams_ratio_summary_short_parent.csv
Calibration/M3G20081211_RDN_BPF.csv
```

The RELAB points table is large, so it is scanned once for all required
measurements rather than once per figure.

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
  relationship type, counts, and complete lists of included RELAB measurement
  and specimen IDs;
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
