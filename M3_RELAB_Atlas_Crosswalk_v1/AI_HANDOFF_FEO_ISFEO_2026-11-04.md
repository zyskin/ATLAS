# AI research handoff: M3 reflectance to FeO and Is/FeO (2026-11-04)

## Mission
Extend and independently evaluate the existing Apollo/RELAB-to-M3 forward-modelled spectral inference pipeline. Predict **FeO (wt.%)** and **Is/FeO (maturity index)**; compare existing Lucey/Kumar approaches with modified physical/statistical and new interpretable ML methods, and test robustness on actual orbital M3 pixels. Do not declare an orbital calibration based on laboratory cross-validation alone.

## Repository and provenance
- GitHub: https://github.com/zyskin/ATLAS
- Default branch: `main`
- Relevant root: `M3_RELAB_Atlas_Crosswalk_v1/`
- Previous unmerged work was merged in PR https://github.com/zyskin/ATLAS/pull/4 on 2026-11-04 (merge SHA `997ea78d461a1cfc31b49142a32a1b91c912a1e4`).
- Clone: `git clone https://github.com/zyskin/ATLAS.git && cd ATLAS/M3_RELAB_Atlas_Crosswalk_v1`
- Previously reported user-local project location (not portable): `/media/maxim/Elements/grants/2024/INSRU/pgdal/M3_RELAB_Atlas_Crosswalk_v1`.
- Previously used M3 snapshot: `M3G20090107T011405`; earlier band tables and associated images/angles were in `/media/maxim/Elements/grants/2024/INSRU/pgdal/29_31_m31_p30`. Check actual files and headers locally before processing. These absolute paths are not GitHub storage locations.

## Critical inputs and provenance
- `data/ml_combined_v2/m3_feo_isfeo_ground_truth_v2.csv`: 524 RELAB/M3-basis spectral rows, 74 spectral features `m3_b01` ... `m3_b74`, specimen/parent identifiers, direct and provisional FeO targets, Is/FeO target(s), baseline predictions. Inspect actual CSV column names instead of guessing the Is/FeO target column.
- `data/ml_combined_v2/m3_feo_isfeo_claims_v2.csv`: unique source claims / provenance; check which spectra inherited parent chemistry.
- `data/ml_combined_v2/m3_feo_isfeo_parent_summary_v2.csv`: aggregated Apollo parent groups.
- `data/ml_combined_v2/m3_feo_isfeo_ground_truth_v2_manifest.json`: input checksums, exclusions, sources, constants, warnings.
- `data/source_extracts/`: JSC 19069 lunar soils and LPI regolith breccia extract tables and `source_chemistry_index.csv`; read `data/source_extracts/README.md`.
- Earlier source data include `relab_raw_native_2026-10-04/atlas_relab_feo_table.csv`, `data/relab_m3_v1/relab_m3_spectra.csv`, `data/structured_v1/atlas_feo_measurements.csv`, `data/ai_resolution_v1/`, `m3_feo_ml_preliminary/`, and `data/is_feo_v1/`.
- `data/is_feo_v1/FEO_IS_FEO_STATUS.md`: caution on association and provenance. **Is/FeO is ferromagnetic resonance intensity normalized by FeO**, not a spectral ratio.
- FeO zero for Apollo sample 12024 was a placeholder and is excluded; do not restore it as a measurement. Never silently use graph-read values, Bhatt-derived rows, or unvalidated machine-extracted candidates as ground truth. Parent-to-child matching is provisional and must remain explicitly labeled, never treated as an exact chemical measurement.

## Actual dataset coverage
- 524 usable RELAB measurements, 108 Apollo parent groups.
- FeO combined target: 523 spectra / 107 parent groups.
- Direct specimen-associated FeO only: 105 spectra / 33 groups.
- Is/FeO: 113 spectra / 29 groups; smaller independent grouping sample. Inspect target completeness and direct-versus-parent association first.
- 456 unique retained chemical claims: 427 FeO and 29 Is/FeO.
- Across source extracts: JSC 19069 has 169 lunar soil rows, 132 FeO and 157 Is/FeO; breccia workbook 45 sample rows, 42 FeO rows.
- Distinguish 66 direct-only rows, 39 mixed exact-plus-parent rows, 418 provisional-parent rows and 1 Is/FeO-only row (classification as reported in v2 documentation).

## Where the code is
- `m3_feo_ml_v2/build_combined_ground_truth_v2.py`: joins source claims, RELAB spectra, parent/child identifiers, Lucey/Kumar quantities, and writes the combined dataset.
- `m3_feo_ml_v2/train_combined_feo_models_v2.py`: earlier grouped FeO comparison.
- `m3_feo_ml_v2/README.md`: exact source rebuild commands and v2 explanations.
- `m3_feo_ml_v3/train_compare_interpret_feo_models_v3.py`: current main trainer for FeO (Ridge, Elastic Net, PCR, PLS, RBF SVR, Extra Trees, Random Forest, Gradient Boosting, Gaussian Process), nested parent-grouped CV, coefficient and importance analyses, saved fit artifacts.
- `m3_feo_ml_v3/apply_feo_models_to_m3_v3.py`: inference on M3 stage-1 band TSVs or input spectral CSV with domain flags.
- `m3_feo_ml_v3/M3_FeO_Machine_Learning_Methods.tex`: mathematical documentation; review it critically, define notation and assumptions explicitly, and improve explanations without conflating methods.
- `m3_feo_ml_v3/README.md`: exact commands, list of emitted artifacts, model/domain-warning details.
- `m3_feo_ml_v3/requirements.txt`: Python package requirements.

## Reproduce current runs
From `M3_RELAB_Atlas_Crosswalk_v1/`:
```bash
python -m pip install -r m3_feo_ml_v3/requirements.txt
DATA=data/ml_combined_v2/m3_feo_isfeo_ground_truth_v2.csv
python m3_feo_ml_v3/train_compare_interpret_feo_models_v3.py --dataset "$DATA" --target-column feo_wt_percent_mean --output-dir data/ml_interpretable_v3/reproduced_combined --search-level quick
python m3_feo_ml_v3/train_compare_interpret_feo_models_v3.py --dataset "$DATA" --target-column feo_direct_wt_percent_mean --output-dir data/ml_interpretable_v3/reproduced_direct --search-level quick
```
The `standard` search is larger. Do not fit preprocessing before the outer split. The nested **inner** folds are for hyperparameter selection among the outer-training Apollo parents; each **outer** fold tests unseen Apollo parents. Evaluate parent-equal-weight metrics primarily; report spectrum-weighted metrics separately.

## Previously committed results
- `data/ml_combined_v2/RESULTS_SUMMARY.md`; `data/ml_combined_v2/results_{combined,direct_only}_quick/`: v2 metrics, per-measurement/parent CV predictions, parameters, feature importance, plots, training manifests.
- `data/ml_interpretable_v3/results_{combined,direct_only}_quick/cv_metrics.csv`, `training_manifest.json`, `deployment_manifest.json`: v3 stored summaries.
- The fitted `model_*.joblib` and screening-reference joblibs are **not committed**; regenerate locally. Some detailed v3 interpretation outputs named in v3 README may similarly require regeneration. Do not imply those binaries or all detailed CSVs are available on GitHub.
- v3 parent-group MAE in wt.% FeO (combined/direct-only): Gaussian Process 2.464/3.717; Extra Trees 2.723/3.226; Random Forest 2.894/3.338; Gradient Boosting 3.000/3.179; RBF SVR 3.149/4.066; PLS 3.581/4.711; Ridge 3.830/4.642; PCR 3.938/4.626; Elastic Net 4.102/4.691; Kumar power principal 2.626/3.498 **on fewer parents**.
- v2 parent-group Lucey baseline combined MAE: standard with quadrant-aware atan2 modulo 2π 3.874; standard principal branch 6.575; saved origin-changed formula about 6.46–6.49. Quadrant/branch choices are scientifically consequential.
- Results are provisional: model rankings change sharply when only direct FeO associations are admitted.

## Lucey/Kumar details
- Standard Lucey reference origin in current scripts: `(0.08, 1.19)`; saved project-fit origin: `(0.03433418510481713, 1.2571662682003644)`; the latter is NOT a published universal calibration.
- Two angle conventions must be kept distinct: `-atan((y-y0)/(x-x0))` (principal branch), and `(-atan2(y-y0,x-x0)) mod 2π` (quadrant-aware).
- Kumar includes three-band averages around 750 and 950 nm and variants; verify actual implementation/validity domains, especially negative principal H. Some Kumar predictions are undefined; publish coverage along with error, never silently drop unpredicted cases.
- Inspect the actual code/math for exact reflectance transforms and coefficients; do not infer formulas from abbreviated prose.

## Applying to M3
M3 orbital observations have RFL reflectance, LOC location/elevation, and OBS geometry files with an associated HDR. Shape varies by snapshot and band indexing must be verified. The existing stage-1 outputs use 74 files `PREFIX_band_1.tsv` ... `PREFIX_band_74.tsv`, each with per-pixel reflectance, OR a CSV with columns `m3_b01` ... `m3_b74`. The primary model uses the M3-forward-modeled RELAB band basis; confirm compatibility of wavelength calibration, band ordering, units and reflectance scaling.
Example:
```bash
python m3_feo_ml_v3/apply_feo_models_to_m3_v3.py --stage1-dir /path/to/stage1_output --prefix M3G20090107T011405 --bundle-dir data/ml_interpretable_v3/reproduced_combined --output-dir /path/to/m3_feo_predictions --chunk-lines 64
```
Bitmask domain flags: 1 invalid spectrum, 2 reflectance out of broad range, 4 bands outside training range, 8 PCA/Mahalanobis departure, 16 negative FeO prediction, 32 above configured FeO cap, 64 large inter-model disagreement. Zero means no implemented flag, **not** proven physical correctness. Preserve un-clipped predictions and warnings; no physical clipping as a substitute for validation.

## Research tasks / desired implementation
1. Reproduce/check committed v2/v3 results and group splits; record versions, seeds, SHA256 input hashes, precise command lines and numerical differences.
2. Create **Is/FeO ML training and baselines** rather than assuming FeO trainer already handles the target perfectly. Inspect the available 113 spectra / 29 parents and association levels; evaluate whether log transforms (strictly positive only), robust regression, multi-task modeling with FeO, maturity-sensitive continuum/slope/absorption features, and physics-motivated corrections are justified. Never pretend Is/FeO is itself a reflectance quotient.
3. Investigate modified Lucey/Kumar feature mappings with grouped out-of-fold fitting of origin, coefficients and regularization; use scientifically interpretable transformations; avoid fitting calibrations on held-out parents.
4. Compare new models (e.g. constrained GAM/splines, monotonic or partially monotonic boosting where defensible, robust Huber, Bayesian ridge, kernel methods, spectral derivative and band-ratio features, Gaussian process with spectral kernels, ensemble/stacking with strictly nested out-of-fold training). Prioritize interpretable models and appropriately sized searches for 29 independent Is/FeO parents.
5. Quantify spectral-band contributions with standardized linear coefficients, grouped-band permutation, SHAP or equivalent where appropriate, partial dependence/ICE, and local derivatives. Adjacent spectral bands are strongly correlated, so single-band importance is not uniquely attributable.
6. Test hard distribution shift: cross-validation on **Apollo parent IDs**, sensitivity for exact-versus-parent-matched FeO, Apollo mission or site holdouts when feasible, wavelength/preprocessing perturbations, illumination/scaling stress tests, and empirical application to actual M3 snapshot(s).
7. For actual M3 outputs report flag rates, prediction distributions, spatial structure, disagreement, regional extremes, training-spectral nearest-neighbor distances and physically impossible values *before clipping*. Compare against independent compositional maps/sites where scientifically defensible and note differences in spatial resolution.
8. Compare parent-level MAE, RMSE, bias, R², sample counts, fold/coverage, missingness; for Is/FeO choose meaningful scales/units and avoid interpreting small-sample R² alone. Supply grouped bootstrap uncertainty estimates, with the parent as sampling unit.
9. Deliver runnable code, tests, CSV predictions per spectrum and parent, selected hyperparameters, model binaries + regeneration instructions, plots, interpretation tables, provenance/caveat updates, and a concise math report defining objectives, transformations, inner/outer CV, model selection and extrapolation screening.
10. Add work in separately reviewable commits or a PR; don't rewrite source claims or overwrite prior result versions. Keep a changelog and document exact paths for all new artifacts.

## First inspection commands
```bash
git status --short
find m3_feo_ml_v2 m3_feo_ml_v3 data/ml_combined_v2 data/ml_interpretable_v3 -maxdepth 3 -type f | sort
python - <<'PY'
import pandas as pd
d=pd.read_csv('data/ml_combined_v2/m3_feo_isfeo_ground_truth_v2.csv')
print('shape',d.shape)
for c in d.columns:
    if 'feo' in c.lower() or 'parent' in c.lower() or 'is_' in c.lower():
        print(c, d[c].notna().sum())
print('spectral bands:',[c for c in d if c.startswith('m3_b')])
PY
```

## First concrete deliverable
Implement an independent **Is/FeO grouped-CV baseline and interpretable model comparison** (with exact provenance and appropriate target transforms), plus a new FeO calibration/regularization experiment, and a stress-tested M3 application report. Show methods and formulas with precise notation before interpretation; never substitute a cookbook summary for a mathematical definition. State clearly what is replicated, what is newly tested, and what remains unvalidated.
