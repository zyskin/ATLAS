# Combined ground truth and quick FeO regression results

Generated 2026-11-02.

## Coverage

- 524 RELAB/M3-basis spectral records from 108 Apollo parents.
- FeO: 523 records from 107 parents.
- Is/FeO: 113 records from 29 parents.
- 456 unique retained claims: 427 FeO and 29 Is/FeO.
- 105 spectra from 33 parents have FeO explicitly associated with the RELAB
  measurement; the remaining FeO training rows use Apollo-parent chemistry.

## Primary quick-run comparison

The primary unit below is the held-out Apollo parent.  Predictions and targets
are averaged within each parent before scoring.

| Method | Parents scored | MAE (wt.% FeO) | RMSE | R2 |
|---|---:|---:|---:|---:|
| Kumar power, principal angle | 101 | 2.626 | 3.486 | 0.664 |
| Extra Trees | 107 | 2.726 | 3.717 | 0.629 |
| RBF SVR | 107 | 3.148 | 4.437 | 0.471 |
| Standard Lucey, atan2 mod 2pi | 107 | 3.874 | 7.223 | -0.401 |
| Ridge | 107 | 3.933 | 4.720 | 0.402 |
| PLS | 107 | 4.028 | 4.854 | 0.367 |
| Origin-changed Lucey, atan2 mod 2pi | 107 | 6.458 | 7.450 | -0.491 |
| Origin-changed Lucey, principal angle | 107 | 6.488 | 7.459 | -0.495 |
| Standard Lucey, principal angle | 107 | 6.575 | 13.507 | -3.901 |

Kumar power has the lowest MAE, but it is undefined for six parent means with
negative principal H and therefore is not a complete-coverage comparison.
Extra Trees is the strongest full-coverage result in this quick run.

## Exact-association sensitivity run

This smaller run uses only FeO directly associated with the RELAB measurement:
105 spectra from 33 Apollo parents.

| Method | Parents scored | MAE | RMSE | R2 |
|---|---:|---:|---:|---:|
| Extra Trees | 33 | 3.143 | 4.647 | 0.352 |
| Kumar power, principal angle | 31 | 3.495 | 4.473 | 0.415 |
| Standard Lucey, atan2 mod 2pi | 33 | 3.845 | 5.120 | 0.214 |
| Ridge | 33 | 4.641 | 5.845 | -0.025 |
| RBF SVR | 33 | 4.894 | 6.727 | -0.358 |
| Origin-changed Lucey, principal angle | 33 | 5.903 | 6.857 | -0.411 |

The sensitivity run is materially weaker.  This shows that the larger result
depends strongly on provisional parent-level chemistry and should not yet be
treated as a validated orbital FeO model.

## Immediate interpretation

1. The arctangent convention matters substantially.  Standard Lucey with
   `atan2 mod 2pi` is much less bad than the principal-angle direct version on
   this data.
2. The saved origin-changed calibration does not transfer well to this expanded
   laboratory set.
3. Extra Trees is the best complete-coverage quick model, but its score drops on
   the strict exact-association subset.
4. The next scientific improvement is not a larger algorithm search.  It is
   resolving parent-versus-fraction chemistry and adding more exact
   spectrum/chemistry pairs, with maturity and grain size retained.
