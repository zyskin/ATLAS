# Bounded FeO regression for orbital M3 deployment (v5)

This directory adds FeO regressors whose output is physically bounded as a mass
fraction.  It compares the legacy v3 model families and the v4/Claude families
on the Apollo/RELAB laboratory dataset, using Apollo-parent-grouped
cross-validation.

## Physical target links

The repository stores FeO in wt.%. Define the mass fraction

`f = FeO(wt.%)/100`.

Two links are implemented.

### Logistic link

`z = log(f/(1-f))`

`f_hat = 1/(1 + exp(-z_hat))`

This guarantees `0 < f_hat < 1` for every input spectrum.

### Positive-tanh link

To implement the proposed form `f = tanh(theta)` with `theta > 0`, the
unconstrained regressor predicts `z` on the real line and the deployment
mapping is

`theta = softplus(z) > 0`

`f_hat = tanh(theta)`.

Training uses the inverse mapping
`z = softplus_inverse(atanh(f))`.

No post-hoc clipping is used.

## Data and validation

Input:
`data/ml_combined_v2/m3_feo_isfeo_ground_truth_v2.csv`

- Combined FeO: 523 spectra / 107 Apollo parents.
- Direct-only FeO: 105 spectra / 33 Apollo parents.
- Input spectra: 74 M3-response-convolved reflectance bands
  `m3_b01 ... m3_b74`.
- Outer and inner validation are grouped by Apollo parent.
- Hyperparameters are selected by Apollo-parent MAE after predictions are
  transformed back to the original FeO wt.% scale.

## Headline Apollo results

Combined FeO, 107 parents:

| bounded model | link | parent MAE (wt.%) | RMSE | R2 | OOF prediction range (wt.%) |
|---|---|---:|---:|---:|---:|
| HistGradientBoosting, physical+raw | logistic | 2.244 | 3.268 | 0.713 | 0.51--22.49 |
| ExtraTrees, physical+raw | positive-tanh | 2.262 | 3.353 | 0.698 | 0.62--20.97 |
| ExtraTrees, physical+raw | logistic | 2.273 | 3.339 | 0.701 | 0.63--21.03 |

For comparison, the previous unbounded v4 ExtraTrees result was MAE 2.292
wt.%.  Thus the bounded output does not show an accuracy penalty in this
partition.

Direct-only FeO, 33 parents:

| bounded model | link | parent MAE (wt.%) | RMSE | R2 | OOF prediction range (wt.%) |
|---|---|---:|---:|---:|---:|
| ExtraTrees, physical+raw | positive-tanh | 3.008 | 4.542 | 0.381 | 0.23--19.27 |
| ExtraTrees, physical+raw | logistic | 3.031 | 4.547 | 0.380 | 0.23--19.32 |

The previous unbounded v4 ExtraTrees result was MAE 3.156 wt.%.  The previous
unbounded v4 KernelRidge result (2.785 wt.%) remains nominally better on the
small direct-only set, but bounded KernelRidge degrades strongly; model ranking
therefore depends on the output parameterization.

All full results are in
`data/ml_bounded_v5/bounded_model_comparison.csv` and
`data/ml_bounded_v5/RESULTS.md`.

## Important distinction: physical versus lunar-plausible

The links guarantee only the generic compositional constraint (0<f<1),
equivalently 0--100 wt.% FeO.  Some flexible bounded regressors can still
produce lunar-implausible values while satisfying that mathematical bound.
For example, several Gaussian-process / spline / angle models produced held-out
predictions above 50 wt.%.

The leading tree models did not show that problem on Apollo held-out data:
their OOF ranges stay near 0--23 wt.%.  For orbital deployment, ExtraTrees is a
conservative first choice because its earlier v4 validation also showed strong
split stability; the bounded HistGradientBoosting model is a strong secondary
candidate and has the best single-partition MAE here.

## Saved deployment models

Every run contains a fitted `final_model.joblib`.  Particularly useful models:

`data/ml_bounded_v5/feo_wt_percent_mean__all/logit_fraction/v4_hist_gbm/final_model.joblib`

`data/ml_bounded_v5/feo_wt_percent_mean__all/logit_fraction/v4_extra_trees/final_model.joblib`

`data/ml_bounded_v5/feo_wt_percent_mean__all/tanh_softplus_fraction/v4_extra_trees/final_model.joblib`

The object exposes:

`predict_fraction(bands)` -- returns FeO fraction in (0,1)

`predict_wt_percent(bands)` -- returns the same prediction in wt.%

where `bands` is an n x 74 array in the same M3 band basis used by training.

## Apply to an M3-basis CSV

If the orbital preprocessing has produced a CSV containing
`m3_b01 ... m3_b74`:

```bash
python -m m3_feo_ml_v5.apply_bounded_feo \
  --model data/ml_bounded_v5/feo_wt_percent_mean__all/logit_fraction/v4_extra_trees/final_model.joblib \
  --input orbital_m3_74band.csv \
  --output orbital_m3_feo.csv
```

The output includes `feo_fraction`, `feo_wt_percent`, and a bound check.

## Caveat for real orbital M3

The models were trained on laboratory RELAB spectra forward-modelled into the
M3 band basis.  A bounded output prevents impossible composition values, but it
does not remove laboratory-to-orbit domain shift (photometry, calibration,
surface roughness, temperature, geometry, noise, etc.).  Orbital predictions
should therefore be accompanied by spectral-domain / extrapolation diagnostics
and comparison of multiple leading bounded models.
