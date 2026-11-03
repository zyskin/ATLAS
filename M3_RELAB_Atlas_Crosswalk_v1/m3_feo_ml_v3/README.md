# M3 FeO regression: comparison, interpretation, and orbital screening

This directory contains version 3 of the preliminary FeO regression work.  It
fits every candidate statistical model, evaluates every method with Apollo
parent-grouped nested cross-validation, preserves the final fitted estimators,
and records several distinct forms of band interpretation.  It also contains a
separate program for applying the fitted models to real M3 reflectance data
without silently forcing implausible predictions into an allowed interval.

The current experiment is preliminary.  It does not establish that a model
trained on laboratory spectra is calibrated for orbital M3 pixels.

## Programs

`train_compare_interpret_feo_models_v3.py`

- fits Ridge, Elastic Net, principal-component regression (PCR), partial least
  squares (PLS), radial-basis-function support-vector regression (RBF-SVR),
  Extra Trees, Random Forest, Gradient Boosting, and Gaussian-process
  regression;
- evaluates the Lucey and Kumar formula variants already stored in the input
  table;
- keeps every Apollo parent entirely within one validation fold;
- selects hyperparameters in an inner Apollo-parent-grouped cross-validation;
- writes held-out predictions, metrics, final fitted models, selected
  hyperparameters, linear coefficients, component matrices, support vectors,
  tree importances, local SVR derivatives, and held-out permutation
  importance;
- constructs a training-domain reference for screening orbital spectra.

`apply_feo_models_to_m3_v3.py`

- reads either a CSV containing `m3_b01` through `m3_b74`, or a Stage-1
  extraction directory containing 74 tall TSV band tables;
- predicts with every saved model;
- preserves raw predictions rather than clipping them;
- records flags for non-finite inputs, broad physical reflectance violations,
  excessive departure from the training range, excessive principal-component
  distance, negative or excessive predicted FeO, and excessive inter-model
  disagreement;
- writes a screened consensus only where no flag is present.

## Reproduce the two completed comparisons

From the project directory:

```bash
ATLAS=/media/maxim/Elements/grants/2024/INSRU/pgdal/M3_RELAB_Atlas_Crosswalk_v1
DATA="$ATLAS/data/ml_combined_v2/m3_feo_isfeo_ground_truth_v2.csv"
CODE="$ATLAS/m3_feo_ml_v3"
OUT="$ATLAS/data/ml_interpretable_v3"

python "$CODE/train_compare_interpret_feo_models_v3.py" \
  --dataset "$DATA" \
  --target-column feo_wt_percent_mean \
  --output-dir "$OUT/results_combined_quick" \
  --search-level quick

python "$CODE/train_compare_interpret_feo_models_v3.py" \
  --dataset "$DATA" \
  --target-column feo_direct_wt_percent_mean \
  --output-dir "$OUT/results_direct_only_quick" \
  --search-level quick
```

The first run uses the combined provisional FeO target: 523 spectra belonging
to 107 Apollo parents.  The second uses only direct specimen-level FeO values:
105 spectra belonging to 33 Apollo parents.  Their performance estimates must
not be conflated.

### Completed quick-search results

The primary unit below is the mean prediction for a held-out Apollo parent.

| Model | Combined MAE | Combined R² | Direct-only MAE | Direct-only R² |
|---|---:|---:|---:|---:|
| Gaussian process | 2.464 | 0.692 | 3.717 | 0.198 |
| Extra Trees | 2.723 | 0.632 | 3.226 | 0.333 |
| Random Forest | 2.894 | 0.603 | 3.338 | 0.269 |
| Gradient Boosting | 3.000 | 0.560 | 3.179 | 0.303 |
| RBF-SVR | 3.149 | 0.480 | 4.066 | 0.048 |
| PLS | 3.581 | 0.448 | 4.711 | -0.044 |
| Ridge | 3.830 | 0.431 | 4.642 | -0.025 |
| PCR with Ridge | 3.938 | 0.401 | 4.626 | -0.025 |
| Elastic Net | 4.102 | 0.357 | 4.691 | -0.082 |
| Kumar power, principal-angle branch | 2.626 | 0.664 | 3.498 | 0.414 |

The Gaussian process is first on the combined target but not on the
direct-only target.  Extra Trees and Gradient Boosting are more stable across
the two definitions.  Kumar's power expression remains a strong baseline.
This instability is itself a result: the current ranking depends materially on
how FeO is associated with spectra, so no model should yet be declared the
orbital calibration.

Use `--search-level standard` for a larger hyperparameter search.  This is more
computationally expensive and still does not replace external orbital
validation.

## Apply the saved estimators to a Stage-1 M3 snapshot

The following command expects the 74 files
`M3G..._band_1.tsv` through `M3G..._band_74.tsv` in `--stage1-dir`:

```bash
PREFIX=M3G20090107T011405

python "$CODE/apply_feo_models_to_m3_v3.py" \
  --stage1-dir /path/to/stage1_output \
  --prefix "$PREFIX" \
  --bundle-dir "$OUT/results_combined_quick" \
  --output-dir /path/to/feo_predictions_v3 \
  --chunk-lines 64
```

For an ordinary CSV:

```bash
python "$CODE/apply_feo_models_to_m3_v3.py" \
  --input-csv /path/to/m3_spectra.csv \
  --bundle-dir "$OUT/results_combined_quick" \
  --output-dir /path/to/feo_predictions_v3
```

The combined-target bundle is useful for stress testing because it has broader
coverage.  It is not a validated final calibration.  Repeat the application
with the direct-only bundle and compare both the predictions and the domain
flags.

## Output interpretation

The main files are:

- `cv_metrics.csv`: cross-validation metrics by individual measurement and by
  Apollo-parent mean.  The parent-mean rows are the primary comparison.
- `cv_predictions_by_measurement.csv`: honest outer-fold predictions for each
  laboratory spectrum.
- `cv_predictions_by_parent.csv`: the corresponding Apollo-parent averages.
- `best_parameters.json`: selected hyperparameters and every inner-fold search
  score.
- `model_*.joblib`: final estimators fitted to all eligible rows.  These fits
  are for prediction; their training residuals are not performance evidence.
- `linear_band_coefficients.csv`: effective coefficients for Ridge, Elastic
  Net, PCR, and PLS, both in standardized coordinates and in original
  reflectance units.
- `pcr_components.csv` and `pls_components.csv`: component directions and
  associated quantities.
- `svr_support_vectors.csv`: support-vector identities and dual coefficients.
- `svr_band_sensitivity.csv`: average local derivatives with respect to each
  standardized band.
- `tree_feature_importance.csv`: impurity-based importance for tree ensembles.
- `cv_permutation_importance.csv`: increase in held-out MAE after one band is
  permuted.  Adjacent bands are strongly correlated, so single-band
  permutation effects are not unique causal contributions.
- `band_interpretation_ranked.csv`: the same non-equivalent interpretation
  quantities placed in one ranked table with an explicit `measure` column.
- `training_band_statistics.csv`, `training_reference_spectra.csv`, and
  `training_domain_reference.joblib`: the orbital-domain screening reference.
- `deployment_manifest.json`: the model inventory and prediction policy.

## Orbital-screening bit mask

`apply_feo_models_to_m3_v3.py` uses an integer bit mask.  A value of zero means
that none of the implemented checks failed; it does not prove physical
validity.

| Bit | Meaning |
|---:|---|
| 1 | non-finite or otherwise invalid input spectrum |
| 2 | at least one reflectance lies outside the configured broad physical interval |
| 4 | too many bands lie outside the laboratory training minimum and maximum |
| 8 | principal-component/Mahalanobis distance exceeds the configured training threshold |
| 16 | at least one model predicts negative FeO |
| 32 | at least one model predicts FeO above the configured upper limit |
| 64 | the difference between the largest and smallest model predictions is excessive |

Decode a flag with bitwise AND; for example, `flag & 8 != 0` means the
principal-component distance check failed.

## Scientific restrictions

1. Apollo children and size fractions sharing one parent are correlated.  The
   grouped validation prevents direct leakage but does not create more
   independent lunar compositions.
2. The combined target often associates parent-level chemistry with a child
   spectrum.  The direct-only analysis is smaller but more conservative.
3. Laboratory spectra and orbital M3 reflectance differ in illumination,
   viewing geometry, photometric correction, temperature, spatial mixing,
   maturity, grain size, and calibration history.
4. Band coefficients and importance scores describe the fitted statistical
   relation, not a unique mineralogical decomposition.
5. Orbital predictions should be interpreted only after inspecting the flags,
   model disagreement, spatial coherence, and performance on independently
   characterized lunar regions.

