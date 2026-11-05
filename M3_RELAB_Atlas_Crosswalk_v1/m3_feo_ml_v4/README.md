# m3_feo_ml_v4 — new Apollo/RELAB FeO & Is/FeO ML experiments

New (v4) machine-learning code and executed experiments for predicting **FeO
(wt.%)** and **Is/FeO maturity** from M3-response-convolved RELAB laboratory
spectra, with strict **Apollo-parent-grouped** cross-validation. This package is
self-contained and reproducible without the original chat session.

It does **not** touch orbital M3 imagery. All work is on the laboratory ground
truth already in the repo.

## Data

Input (unchanged): `data/ml_combined_v2/m3_feo_isfeo_ground_truth_v2.csv`
- 524 RELAB spectral records, 108 Apollo parent groups.
- 74 M3-convolved reflectance bands: `m3_b01`…`m3_b74` (460.99 … 2537.04 nm;
  band centres from the M3 forward model, `data/relab_m3_v1/relab_m3_spectra.csv`,
  verified band 9 = 750 nm, band 19 = 950 nm).
- Targets:
  - `feo_wt_percent_mean` — combined/provisional FeO: **523 spectra / 107 parents**.
  - `feo_direct_wt_percent_mean` — strictly directly-associated FeO:
    **105 spectra / 33 parents**.
  - `is_feo_mean` — Is/FeO ferromagnetic-resonance maturity (**not** a spectral
    quotient): **113 spectra / 29 parents**, values 1–90 (strictly positive, so
    `log1p` is admissible).

**Grouping key:** `i_atlas_sample` (Apollo parent). One parent never appears in
both train and test of any split. Many spectra share a parent and repeat
chemistry values, so all scoring is reported at the **parent-mean** level as the
primary metric.

No zero-imputation of missing targets; child and parent chemistry are never
silently equated; no target-derived Lucey/Kumar prediction columns are used as
input features (they are scored only as fixed reference baselines).

## Design

- **Single configurable trainer** (`trainer.py`): one function runs a fully
  specified experiment = (target column, source-quality subset, feature family,
  model family, target transform, seed, outer/inner grouped folds, output dir).
- **Nested grouped CV**: `GroupKFold` outer folds for honest evaluation, inner
  `GroupKFold` for hyper-parameter tuning by parent-mean MAE. **All**
  preprocessing (scaling, splines) and tuning happen inside each training fold.
- **Target transforms** (`none`, `log1p`): the inverse is applied before scoring,
  on the original target scale.
- **Feature families** (`features.py`), all computed per-row from reflectance and
  the known band wavelengths (no data-derived parameters ⇒ leak-free):
  - `raw` — the 74 bands.
  - `physical` — compact descriptors: VIS–NIR and short-VIS log-slopes, band
    means / ratios / log-ratios (R950/R750 etc.), a 1 µm ferrous band depth with
    linear continuum removal and a band-centre position proxy, 2 µm shoulder
    ratio, visible contrasts, overall reflectance level.
  - `derivative` — 1st/2nd spectral derivatives on a coarse grid.
  - `shape` — continuum-removed (albedo-insensitive) shape + raw bands.
  - `phys_raw` — physical descriptors + raw bands.
- **Model families** (`models.py`): Huber, Bayesian Ridge (both also with
  monotone **spline** transforms of the compact physical features), Ridge,
  **HistGradientBoosting** (shallow trees, leaf/`l2` restriction for ~100
  parents), **KernelRidge (RBF)**, **Gaussian Process** (Constant·RBF + explicit
  WhiteKernel, bounded length-scale — a genuinely different kernel from v3),
  Extra Trees, and a `null_mean` dummy.
- **Modified Lucey/Kumar** (`SpectralAngleRegressor`): the ferrous spectral-angle
  model re-derived from scratch (R750/R950 band means → angle about an origin →
  polynomial), with origin and/or coefficients fitted **only on training
  parents** (inner CV chooses degree and whether to fit the origin). Compared
  against the fixed published formula columns.
- **Interpretation**: standardized coefficients for linear models; **leak-free
  block-permutation importance** (permute a block of *adjacent* bands / a feature
  group in the *held-out* parents of each fold, measure parent-MAE increase).
- **Outputs per experiment**: `oof_predictions.csv` (row id, Apollo parent,
  scope, target, OOF prediction, fold), `oof_parent_means.csv`,
  `fold_metrics.csv`, `block_permutation_importance.csv`,
  `linear_standardized_coefficients.csv` (linear models), `final_model.joblib`
  (refit on all usable rows), and `manifest.json` (config, fold params, metrics,
  dataset SHA-256, library versions).

## Reproduce

Run from `M3_RELAB_Atlas_Crosswalk_v1/`. Requires `scikit-learn`, `numpy`,
`pandas`, `scipy`, `joblib`, `matplotlib`.

Smoke test (synthetic grouped data, a few seconds):

```bash
python -m m3_feo_ml_v4.smoke_test
```

Full experiment matrix (FeO combined, FeO direct-only, Is/FeO):

```bash
python -m m3_feo_ml_v4.run_experiments \
  --dataset data/ml_combined_v2/m3_feo_isfeo_ground_truth_v2.csv \
  --base-dir data/ml_experiments_v4
```

Single experiment (example):

```bash
python -m m3_feo_ml_v4.train_single \
  --dataset data/ml_combined_v2/m3_feo_isfeo_ground_truth_v2.csv \
  --target-column feo_wt_percent_mean --model bayesian_ridge_spline \
  --feature-family physical --outer-folds 5 --inner-folds 3 \
  --output-dir data/ml_experiments_v4/feo_combined/bayesian_ridge_spline_physical
```

Comparison tables + figures, and robustness stress tests:

```bash
python -m m3_feo_ml_v4.analyze --base-dir data/ml_experiments_v4
python -m m3_feo_ml_v4.robustness \
  --dataset data/ml_combined_v2/m3_feo_isfeo_ground_truth_v2.csv \
  --base-dir data/ml_experiments_v4
```

## Validation stage

`VALIDATION_RESULTS.md` reports a robustness stage (outputs in
`data/ml_experiments_v4_validation/`, originals untouched):

```bash
python -m m3_feo_ml_v4.bootstrap                         # parent-bootstrap CIs + paired ΔMAE
python -m m3_feo_ml_v4.validation --base-dir data/ml_experiments_v4_validation
python -m m3_feo_ml_v4.write_validation_results
```

Key corrections/findings there: (1) the block-permutation metric was fixed to be
**parent-level** (both parent- and measurement-level now saved); the 1 µm ferrous
region still dominates FeO. (2) Paired Apollo-parent bootstraps show the top
models are **statistically indistinguishable** (ΔMAE CIs include zero), while all
Is/FeO models beat the null. (3) A feature ablation shows the FeO gain comes from
the physical features, not changed tree settings. (4) An `log1p`-target Is/FeO GP
removes the unphysical negative predictions.

## Results

See `RESULTS.md` (written by the executed run) and
`data/ml_experiments_v4/comparison_v3_v4_*.csv`. Headline numbers and the full
leaderboard are summarised there. Metrics are parent-grouped out-of-fold; every
numeric claim comes from saved validation outputs, never training fit.

## Caveats

- Is/FeO has only **29 parents**; results are indicative, regularisation is
  prioritised over search, and uncertainty is wide.
- `feo_wt_percent_mean` includes provisional parent-level chemistry; the strict
  `feo_direct_wt_percent_mean` (33 parents) is the conservative setting.
- Predictive band associations are **not** causal mineralogy.
- No orbital validation is claimed.
