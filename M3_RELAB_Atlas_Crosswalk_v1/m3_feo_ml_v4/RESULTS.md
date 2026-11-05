# v4 results — Apollo/RELAB FeO & Is/FeO (parent-grouped CV)

All metrics are **out-of-fold, Apollo-parent-grouped** (primary unit = Apollo-parent mean). `v4_ml_model` = new models from this package; `v4_fixed_published_formula` / `v4_null` = baselines scored on matched parents; `v3_stored` = metrics from `data/ml_interpretable_v3`. Fixed-formula rows with `n_parents < total` have partial coverage.

## FeO combined (`feo_wt_percent_mean`, 107 parents)

**Best new model: `extra_trees_phys_raw`** — parent MAE 2.292, RMSE 3.276, R² 0.712 (n=107 parents).

| rank | method | source | parent MAE | parent RMSE | parent R² | n_parents |
|---:|---|---|---:|---:|---:|---:|
| 1 | extra_trees_phys_raw | v4_ml_model | 2.292 | 3.276 | 0.712 | 107 |
| 2 | kernel_ridge_rbf_raw | v4_ml_model | 2.330 | 3.326 | 0.703 | 107 |
| 3 | hist_gbm_phys_raw | v4_ml_model | 2.376 | 3.478 | 0.675 | 107 |
| 4 | bayesian_ridge_spline_physical | v4_ml_model | 2.378 | 3.180 | 0.728 | 107 |
| 5 | gaussian_process | v3_stored | 2.464 | 3.384 | 0.692 | 107 |
| 6 | kumar_power_principal | v3_stored | 2.626 | 3.486 | 0.664 | 101 |
| 7 | kumar_feo_power_principal | v4_fixed_published_formula | 2.626 | 3.486 | 0.664 | 101 |
| 8 | huber_spline_physical | v4_ml_model | 2.627 | 3.543 | 0.663 | 107 |
| 9 | extra_trees | v3_stored | 2.723 | 3.702 | 0.632 | 107 |
| 10 | hist_gbm_raw | v4_ml_model | 2.795 | 3.546 | 0.662 | 107 |
| 11 | gaussian_process_raw | v4_ml_model | 2.810 | 3.919 | 0.587 | 107 |
| 12 | random_forest | v3_stored | 2.894 | 3.843 | 0.603 | 107 |
| 13 | gradient_boosting | v3_stored | 3.000 | 4.048 | 0.560 | 107 |
| 14 | svr_rbf | v3_stored | 3.149 | 4.398 | 0.480 | 107 |
| 15 | huber_physical | v4_ml_model | 3.223 | 4.190 | 0.528 | 107 |
| 16 | bayesian_ridge_physical | v4_ml_model | 3.234 | 4.099 | 0.549 | 107 |
| 17 | pls | v3_stored | 3.581 | 4.534 | 0.448 | 107 |
| 18 | spectral_angle_modified | v4_ml_model | 3.596 | 4.751 | 0.394 | 107 |
| 19 | kumar_feo_linear_atan2_2pi | v4_fixed_published_formula | 3.647 | 6.395 | -0.098 | 107 |
| 20 | kumar_linear_atan2_2pi | v3_stored | 3.647 | 6.395 | -0.098 | 107 |
| 21 | ridge | v3_stored | 3.830 | 4.602 | 0.431 | 107 |
| 22 | lucey_standard_atan2_2pi | v3_stored | 3.874 | 7.223 | -0.401 | 107 |
| 23 | lucey_standard_feo_atan2_2pi | v4_fixed_published_formula | 3.874 | 7.223 | -0.401 | 107 |
| 24 | pcr_ridge | v3_stored | 3.938 | 4.722 | 0.401 | 107 |
| 25 | ridge_raw | v4_ml_model | 3.962 | 4.736 | 0.398 | 107 |
| 26 | kumar_feo_power_atan2_2pi | v4_fixed_published_formula | 4.015 | 7.754 | -0.615 | 107 |
| 27 | kumar_power_atan2_2pi | v3_stored | 4.015 | 7.754 | -0.615 | 107 |
| 28 | elastic_net | v3_stored | 4.102 | 4.894 | 0.357 | 107 |
| 29 | grouped_mean_null_baseline | v4_nan | 5.324 | 6.342 | -0.080 | 107 |
| 30 | null_mean | v4_ml_model | 5.324 | 6.342 | -0.080 | 107 |
| 31 | kumar_feo_linear_principal | v4_fixed_published_formula | 5.481 | 10.510 | -1.967 | 107 |
| 32 | kumar_linear_principal | v3_stored | 5.481 | 10.510 | -1.967 | 107 |
| 33 | training_parent_mean | v3_stored | 5.537 | 6.259 | -0.052 | 107 |
| 34 | lucey_origin_changed_atan2_2pi | v3_stored | 6.458 | 7.450 | -0.491 | 107 |
| 35 | lucey_origin_changed_feo_atan2_2pi | v4_fixed_published_formula | 6.458 | 7.450 | -0.491 | 107 |
| 36 | lucey_origin_changed_feo_principal | v4_fixed_published_formula | 6.488 | 7.459 | -0.495 | 107 |
| 37 | lucey_origin_changed_principal | v3_stored | 6.488 | 7.459 | -0.495 | 107 |
| 38 | lucey_standard_feo_principal | v4_fixed_published_formula | 6.575 | 13.507 | -3.901 | 107 |
| 39 | lucey_standard_principal | v3_stored | 6.575 | 13.507 | -3.901 | 107 |


**Robustness (extra_trees_phys_raw):**

- Negative predictions: 0 (0.0% of spectra).
- Reflectance perturbation: +2% multiplicative noise shifts parent-mean prediction by 0.827 (units of target); +0.005 additive by 1.012.
- Source-quality breakdown — exact_relab_association_only: MAE 2.443 (n=19); mixed_exact_and_parent: MAE 2.589 (n=14); parent_level_only: MAE 2.299 (n=82).
- Worst parent: 61221 (target 19.660, pred 5.396, |err| 14.264).

## FeO strict direct-only (`feo_direct_wt_percent_mean`, 33 parents)

**Best new model: `kernel_ridge_rbf_raw`** — parent MAE 2.785, RMSE 4.690, R² 0.340 (n=33 parents).

| rank | method | source | parent MAE | parent RMSE | parent R² | n_parents |
|---:|---|---|---:|---:|---:|---:|
| 1 | kernel_ridge_rbf_raw | v4_ml_model | 2.785 | 4.690 | 0.340 | 33 |
| 2 | extra_trees_phys_raw | v4_ml_model | 3.156 | 4.496 | 0.393 | 33 |
| 3 | gradient_boosting | v3_stored | 3.179 | 4.820 | 0.303 | 33 |
| 4 | extra_trees | v3_stored | 3.226 | 4.715 | 0.333 | 33 |
| 5 | random_forest | v3_stored | 3.338 | 4.935 | 0.269 | 33 |
| 6 | hist_gbm_raw | v4_ml_model | 3.472 | 4.992 | 0.252 | 33 |
| 7 | kumar_power_principal | v3_stored | 3.498 | 4.479 | 0.414 | 31 |
| 8 | kumar_feo_power_principal | v4_fixed_published_formula | 3.498 | 4.479 | 0.414 | 31 |
| 9 | hist_gbm_phys_raw | v4_ml_model | 3.553 | 5.220 | 0.183 | 33 |
| 10 | kumar_power_atan2_2pi | v3_stored | 3.643 | 4.732 | 0.328 | 33 |
| 11 | kumar_feo_power_atan2_2pi | v4_fixed_published_formula | 3.643 | 4.732 | 0.328 | 33 |
| 12 | kumar_linear_atan2_2pi | v3_stored | 3.643 | 4.716 | 0.333 | 33 |
| 13 | kumar_feo_linear_atan2_2pi | v4_fixed_published_formula | 3.643 | 4.716 | 0.333 | 33 |
| 14 | gaussian_process | v3_stored | 3.717 | 5.172 | 0.198 | 33 |
| 15 | lucey_standard_atan2_2pi | v3_stored | 3.845 | 5.120 | 0.214 | 33 |
| 16 | lucey_standard_feo_atan2_2pi | v4_fixed_published_formula | 3.845 | 5.120 | 0.214 | 33 |
| 17 | spectral_angle_modified | v4_ml_model | 3.971 | 5.600 | 0.059 | 33 |
| 18 | svr_rbf | v3_stored | 4.066 | 5.633 | 0.048 | 33 |
| 19 | gaussian_process_raw | v4_ml_model | 4.231 | 5.662 | 0.038 | 33 |
| 20 | huber_spline_physical | v4_ml_model | 4.266 | 5.795 | -0.007 | 33 |
| 21 | bayesian_ridge_physical | v4_ml_model | 4.508 | 5.730 | 0.015 | 33 |
| 22 | pcr_ridge | v3_stored | 4.626 | 5.844 | -0.025 | 33 |
| 23 | ridge | v3_stored | 4.642 | 5.846 | -0.025 | 33 |
| 24 | elastic_net | v3_stored | 4.691 | 6.006 | -0.082 | 33 |
| 25 | pls | v3_stored | 4.711 | 5.899 | -0.044 | 33 |
| 26 | ridge_raw | v4_ml_model | 5.045 | 6.524 | -0.277 | 33 |
| 27 | training_parent_mean | v3_stored | 5.053 | 5.831 | -0.020 | 33 |
| 28 | bayesian_ridge_spline_physical | v4_ml_model | 5.198 | 7.543 | -0.707 | 33 |
| 29 | null_mean | v4_ml_model | 5.317 | 6.748 | -0.366 | 33 |
| 30 | grouped_mean_null_baseline | v4_nan | 5.317 | 6.748 | -0.366 | 33 |
| 31 | lucey_origin_changed_feo_atan2_2pi | v4_fixed_published_formula | 5.903 | 6.857 | -0.411 | 33 |
| 32 | lucey_origin_changed_feo_principal | v4_fixed_published_formula | 5.903 | 6.857 | -0.411 | 33 |
| 33 | lucey_origin_changed_principal | v3_stored | 5.903 | 6.857 | -0.411 | 33 |
| 34 | lucey_origin_changed_atan2_2pi | v3_stored | 5.903 | 6.857 | -0.411 | 33 |
| 35 | huber_physical | v4_ml_model | 5.916 | 8.385 | -1.110 | 33 |
| 36 | kumar_feo_linear_principal | v4_fixed_published_formula | 6.302 | 10.985 | -2.621 | 33 |
| 37 | kumar_linear_principal | v3_stored | 6.302 | 10.985 | -2.621 | 33 |
| 38 | lucey_standard_feo_principal | v4_fixed_published_formula | 7.421 | 14.336 | -5.166 | 33 |
| 39 | lucey_standard_principal | v3_stored | 7.421 | 14.336 | -5.166 | 33 |


**Robustness (kernel_ridge_rbf_raw):**

- Negative predictions: 0 (0.0% of spectra).
- Reflectance perturbation: +2% multiplicative noise shifts parent-mean prediction by 0.097 (units of target); +0.005 additive by 0.127.
- Source-quality breakdown — exact_relab_association_only: MAE 2.767 (n=19); mixed_exact_and_parent: MAE 2.809 (n=14).
- Worst parent: 61221 (target 19.660, pred 3.238, |err| 16.422).

## Is/FeO maturity (`is_feo_mean`, 29 parents)

**Best new model: `gaussian_process_raw`** — parent MAE 14.249, RMSE 17.289, R² 0.561 (n=29 parents).

| rank | method | source | parent MAE | parent RMSE | parent R² | n_parents |
|---:|---|---|---:|---:|---:|---:|
| 1 | gaussian_process_raw | v4_ml_model | 14.249 | 17.289 | 0.561 | 29 |
| 2 | extra_trees_phys_raw | v4_ml_model | 15.191 | 18.837 | 0.479 | 29 |
| 3 | kernel_ridge_rbf_raw | v4_ml_model | 15.212 | 19.836 | 0.422 | 29 |
| 4 | hist_gbm_phys_raw | v4_ml_model | 15.609 | 20.190 | 0.401 | 29 |
| 5 | kernel_ridge_rbf_raw_log1p | v4_ml_model | 16.233 | 20.522 | 0.381 | 29 |
| 6 | huber_physical | v4_ml_model | 16.355 | 23.453 | 0.192 | 29 |
| 7 | bayesian_ridge_physical | v4_ml_model | 17.204 | 24.297 | 0.133 | 29 |
| 8 | ridge_physical_log1p | v4_ml_model | 18.042 | 23.395 | 0.196 | 29 |
| 9 | huber_physical_log1p | v4_ml_model | 19.199 | 38.360 | -1.162 | 29 |
| 10 | ridge_physical | v4_ml_model | 19.965 | 27.144 | -0.082 | 29 |
| 11 | grouped_mean_null_baseline | v4_nan | 22.871 | 28.000 | -0.152 | 29 |
| 12 | null_mean | v4_ml_model | 22.871 | 28.000 | -0.152 | 29 |


**Robustness (gaussian_process_raw):**

- Negative predictions: 7 (6.2% of spectra).
- Reflectance perturbation: +2% multiplicative noise shifts parent-mean prediction by 2.565 (units of target); +0.005 additive by 3.414.
- Source-quality breakdown — is_feo_only: MAE 13.148 (n=1); mixed_exact_and_parent: MAE 17.964 (n=5); parent_level_only: MAE 14.761 (n=26).
- Is/FeO predictions outside observed range [1.0, 90.0]: 8 (pred range [-28.7, 93.0]).
- Worst parent: 74220 (target 1.000, pred 38.458, |err| 37.458).

