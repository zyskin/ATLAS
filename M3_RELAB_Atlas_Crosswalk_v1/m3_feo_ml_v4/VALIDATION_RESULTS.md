# v4 validation results — robustness of the reported FeO/Is-FeO improvements

Validation stage for branch `claude/apollo-feo-isfeo-v4`. All metrics are Apollo-parent-grouped. Outputs live under `data/ml_experiments_v4_validation/` and the original `data/ml_experiments_v4/` results are unchanged.

## Headline conclusions

1. **Permutation (corrected, parent-level).** The 1 µm ferrous region (**bands 17–24, ≈930–1050 nm**) remains the top block for FeO in the band-based models (ExtraTrees ΔMAE 1.005; KernelRidge 5.04). The physical-feature model relies on the continuum log-slope + band ratios + 1 µm depth — the same physics. Is/FeO is driven instead by the **broad 1.8–2.4 µm NIR continuum** (and visible), not the 1 µm band, consistent with maturity being a broadband reddening effect. The earlier FeO conclusion **survives** the metric fix.
2. **Rank differences among the top models are NOT statistically significant.** Every paired ΔMAE 95% CI includes zero — e.g. FeO-combined ExtraTrees vs KernelRidge is −0.04 [−0.32, +0.27] (ExtraTrees wins only 61% of resamples). The headline "2.292 vs 2.330 vs 2.378" is within noise. The v4-over-v3 gain is also within uncertainty: v3 GP (2.464) lies **inside** the v4 ExtraTrees MAE CI [1.88, 2.76]. **What is robust:** all three Is/FeO models beat the grouped-mean null (CI excludes 0, P ≥ 0.99).
3. **Randomized-fold stability.** Rankings are broadly stable but some single-partition point estimates were optimistic: FeO-direct KernelRidge 2.785 → mean 2.965 (std 0.34) across partitions; Is/FeO GP 14.25 → mean 13.91 (robust). FeO-combined ExtraTrees is the most stable (2.29 → 2.348, std 0.065).
4. **The FeO gain is genuinely from feature engineering, not tuning.** With identical ExtraTrees machinery and folds: FeO-combined raw-bands-only 2.618 → physical-only 2.320 → phys+raw 2.292. v4 raw-only ExtraTrees (2.618) ≈ v3 ExtraTrees (2.723), so the ~0.4 MAE improvement comes from the physical features. For Is/FeO the feature effect is large: raw 19.28 → physical 14.98.
5. **Is/FeO log-target GP is more physically sensible.** `log1p` GP eliminates all negative predictions (raw GP produced 7) and is **slightly better on average across randomized partitions** (mean 13.91 vs 14.96), though on the single deterministic partition it scored worse (17.46 vs 14.25) and can overshoot high (up to ~150 vs observed max 90). Recommended as the more defensible default, with the high-end overshoot noted.

## 1. Corrected permutation importance (parent-level)

The earlier block-permutation metric used **measurement-level** MAE while the plots/README described **parent-level** MAE. `trainer.py` now computes both; parent-level aggregates predictions and targets by Apollo parent *after* each permutation. Recomputed for the top models (8 permutation repeats/fold, fixed deterministic folds).

| scenario | model | parent MAE | top-3 parent-level blocks (ΔMAE) |
|---|---|---|---|
| feo_combined | extra_trees_phys_raw | 2.292 | m3_bblock_m3_b17_m3_b24=1.005; logslope=0.584; logratio=0.577 |
| feo_combined | kernel_ridge_rbf_raw | 2.330 | m3_bblock_m3_b17_m3_b24=5.037; m3_bblock_m3_b25_m3_b32=4.495; m3_bblock_m3_b65_m3_b72=3.758 |
| feo_combined | bayesian_ridge_spline_physical | 2.378 | logslope=4.857; ratio=2.991; band1um=1.327 |
| feo_direct | kernel_ridge_rbf_raw | 2.785 | m3_bblock_m3_b17_m3_b24=2.652; m3_bblock_m3_b25_m3_b32=2.475; m3_bblock_m3_b57_m3_b64=2.010 |
| isfeo | gaussian_process_raw | 14.249 | m3_bblock_m3_b57_m3_b64=20.952; m3_bblock_m3_b65_m3_b72=15.092; m3_bblock_m3_b09_m3_b16=14.623 |
| isfeo | extra_trees_phys_raw | 15.191 | logratio=2.942; ratio=1.444; m3_bblock_m3_b01_m3_b08=0.552 |

_Interpretation note: whether bands 17–24 (≈930–1050 nm, the 1 µm ferrous band) still dominate FeO under parent-level scoring is read directly from the top-3 blocks above._

## 2. Apollo-parent bootstrap CIs and paired ΔMAE

Cluster bootstrap over Apollo parents (10,000 resamples) on the saved out-of-fold parent-mean predictions; no retraining. v3 saved only aggregate metrics (no per-parent predictions), so it appears only as an **unpaired** reference.

**Per-model parent-MAE/RMSE 95% CI:**

| scenario | model | parent MAE [95% CI] | parent RMSE [95% CI] | bias [95% CI] |
|---|---|---|---|---|
| feo_combined | extra_trees_phys_raw | 2.292 [1.875, 2.760] | 3.276 [2.547, 4.028] | -0.511 [-1.141, 0.084] |
| feo_combined | kernel_ridge_rbf_raw | 2.330 [1.909, 2.813] | 3.326 [2.584, 4.105] | -0.702 [-1.335, -0.108] |
| feo_combined | bayesian_ridge_spline_physical | 2.378 [1.993, 2.794] | 3.180 [2.638, 3.753] | -0.753 [-1.352, -0.171] |
| feo_direct | kernel_ridge_rbf_raw | 2.785 [1.641, 4.205] | 4.690 [2.374, 6.627] | -1.482 [-3.112, -0.090] |
| feo_direct | extra_trees_phys_raw | 3.156 [2.146, 4.317] | 4.496 [2.963, 5.862] | -0.962 [-2.515, 0.492] |
| isfeo | gaussian_process_raw | 14.249 [10.885, 17.828] | 17.289 [13.601, 20.744] | 0.329 [-5.841, 6.664] |
| isfeo | extra_trees_phys_raw | 15.191 [11.485, 19.581] | 18.837 [13.755, 24.434] | 2.789 [-3.858, 9.757] |
| isfeo | kernel_ridge_rbf_raw | 15.212 [10.921, 20.104] | 19.836 [13.811, 25.696] | -1.908 [-9.130, 5.403] |
| isfeo | null_mean | 22.871 [17.101, 28.805] | 28.000 [21.776, 33.403] | 6.727 [-3.189, 16.538] |

**Paired ΔMAE (model_A − model_B; <0 ⇒ A better; identical resampled parents):**

| scenario | A − B | ΔMAE [95% CI] | P(A wins) | CI excl. 0? |
|---|---|---|---|---|
| feo_combined | extra_trees_phys_raw − kernel_ridge_rbf_raw | -0.039 [-0.323, 0.273] | 0.611 | no |
| feo_combined | extra_trees_phys_raw − bayesian_ridge_spline_physical | -0.086 [-0.379, 0.217] | 0.733 | no |
| feo_combined | kernel_ridge_rbf_raw − bayesian_ridge_spline_physical | -0.047 [-0.412, 0.310] | 0.606 | no |
| feo_direct | kernel_ridge_rbf_raw − extra_trees_phys_raw | -0.371 [-1.113, 0.327] | 0.841 | no |
| isfeo | gaussian_process_raw − extra_trees_phys_raw | -0.942 [-3.857, 1.934] | 0.736 | no |
| isfeo | gaussian_process_raw − kernel_ridge_rbf_raw | -0.963 [-4.772, 2.845] | 0.688 | no |
| isfeo | gaussian_process_raw − null_mean | -8.623 [-14.501, -2.792] | 0.999 | yes |
| isfeo | extra_trees_phys_raw − kernel_ridge_rbf_raw | -0.021 [-2.931, 2.677] | 0.492 | no |
| isfeo | extra_trees_phys_raw − null_mean | -7.681 [-12.587, -2.938] | 0.999 | yes |
| isfeo | kernel_ridge_rbf_raw − null_mean | -7.659 [-13.774, -1.438] | 0.993 | yes |

**v3 unpaired reference (point estimates):**

| scenario | v3 method | v3 parent MAE | n_parents |
|---|---|---|---|
| feo_combined | gaussian_process | 2.464 | 107 |
| feo_combined | gradient_boosting | 3.000 | 107 |
| feo_combined | extra_trees | 2.723 | 107 |
| feo_direct | gaussian_process | 3.717 | 33 |
| feo_direct | gradient_boosting | 3.179 | 33 |
| feo_direct | extra_trees | 3.226 | 33 |

## 3. Repeated randomized grouped-CV stability

`RandomizedGroupKFold` shuffles whole Apollo parents into folds with a seeded partition (ordinary `GroupKFold` gives one deterministic, seed-independent split). Inner model selection stays parent-grouped.

| scenario | model | repeats | mean MAE | median | std | min | max |
|---|---|---|---|---|---|---|---|
| feo_combined | bayesian_ridge_spline_physical | 5 | 2.553 | 2.556 | 0.091 | 2.429 | 2.649 |
| feo_combined | extra_trees_phys_raw | 5 | 2.348 | 2.344 | 0.065 | 2.257 | 2.412 |
| feo_combined | kernel_ridge_rbf_raw | 5 | 2.499 | 2.474 | 0.060 | 2.439 | 2.594 |
| feo_direct | extra_trees_phys_raw | 5 | 3.005 | 3.017 | 0.137 | 2.810 | 3.162 |
| feo_direct | kernel_ridge_rbf_raw | 5 | 2.965 | 3.002 | 0.339 | 2.443 | 3.387 |
| isfeo | extra_trees_phys_raw | 10 | 14.437 | 14.607 | 1.117 | 12.582 | 16.737 |
| isfeo | gaussian_process_raw | 10 | 13.909 | 13.670 | 1.172 | 12.377 | 16.165 |
| isfeo | kernel_ridge_rbf_raw | 10 | 13.768 | 13.261 | 1.436 | 12.413 | 16.618 |

## 4. ExtraTrees feature ablation (identical folds + tuning)

Isolates the feature-engineering contribution from model/tuning changes: the same ExtraTrees machinery and grouped folds, varying only the feature family.

| scenario | feature family | parent MAE | parent RMSE | parent R² |
|---|---|---|---|---|
| feo_combined | raw | 2.618 | 3.487 | 0.673 |
| feo_combined | physical | 2.320 | 3.374 | 0.694 |
| feo_combined | phys_raw | 2.292 | 3.276 | 0.712 |
| isfeo | raw | 19.278 | 23.154 | 0.212 |
| isfeo | physical | 14.975 | 19.497 | 0.442 |
| isfeo | phys_raw | 15.191 | 18.837 | 0.479 |

_The stored matrix also has HistGradientBoosting raw (FeO combined MAE 2.795) vs phys_raw (2.376), an independent indication of a feature effect._

## 5. Is/FeO GP: raw target vs log1p target

GP with `log1p` target transform fitted and inverted **inside** each fold (no clipping), vs the raw-target GP.

| variant | parent MAE | RMSE | R² | neg preds | out-of-range | pred min | pred max |
|---|---|---|---|---|---|---|---|
| gaussian_process_raw | 14.249 | 17.289 | 0.561 | 7 | 8 | -28.677 | 93.018 |
| gaussian_process_log1p | 17.464 | 21.912 | 0.295 | 0 | 11 | 6.675 | 150.141 |

**Randomized-repeat stability (10 repeats):**

| variant | repeats | mean MAE | median | std | min | max |
|---|---|---|---|---|---|---|
| gaussian_process_log1p | 10 | 13.914 | 13.940 | 2.081 | 11.140 | 16.996 |
| gaussian_process_raw | 10 | 14.957 | 15.634 | 1.967 | 10.793 | 16.959 |

