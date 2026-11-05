# Claude Code task: IMPLEMENT AND RUN Apollo/RELAB FeO and Is/FeO ML experiments NOW

**Time-limited coding session. Prioritize working code, executed runs, committed results. Do not spend the session planning, literature searching, rendering PDFs, or processing orbital M3 snapshots.**

Repository: https://github.com/zyskin/ATLAS
Default branch: main
Working directory after cloning: `ATLAS/M3_RELAB_Atlas_Crosswalk_v1`
Previous merged work: PR #4, merge commit `997ea78d461a1cfc31b49142a32a1b91c912a1e4`.
Orientation: `AI_HANDOFF_FEO_ISFEO_2026-11-04.md`, `m3_feo_ml_v2/README.md`, `m3_feo_ml_v3/README.md`. **Ignore the real-M3 deployment tasks in the earlier handoff for now.**

## Main input — already on GitHub
`data/ml_combined_v2/m3_feo_isfeo_ground_truth_v2.csv`
- 524 RELAB laboratory spectral records from 108 Apollo parent groups, with 74 M3-response-convolved spectral reflectance values in columns `m3_b01` ... `m3_b74`.
- Combined FeO target: `feo_wt_percent_mean`: 523 spectra/107 parents (some target assignments are provisional parent-level chemistry).
- Strict directly associated FeO target: `feo_direct_wt_percent_mean`: 105 spectra/33 parents.
- Is/FeO: 113 spectra /29 parents. **Inspect the actual CSV to determine exact Is/FeO target, measurement quality, missingness and grouping columns; do not guess a name**. Is/FeO is normalized ferromagnetic resonance maturity, NOT a spectral quotient.
- Provenance: `data/ml_combined_v2/m3_feo_isfeo_claims_v2.csv`, `m3_feo_isfeo_parent_summary_v2.csv`, and ground-truth manifest; `data/source_extracts/`.
- Never impute zero for missing targets or silently equate child and parent specimen chemistry. Do not include target-derived Lucey/Kumar predictions or metadata that leaks target information as input features.
- Check the exact grouping field, one Apollo parent per fold. Beware many spectra per parent and repeated chemistry values.

Existing code:
- `m3_feo_ml_v2/build_combined_ground_truth_v2.py`
- `m3_feo_ml_v2/train_combined_feo_models_v2.py`
- `m3_feo_ml_v3/train_compare_interpret_feo_models_v3.py`: existing Ridge, Elastic Net, PCR, PLS, RBF SVR, Extra Trees, Random Forest, Gradient Boosting, Gaussian Process. Reuse useful classes/evaluation infrastructure; do NOT rewrite existing experiments or spend most time rerunning them.
- `m3_feo_ml_v3/M3_FeO_Machine_Learning_Methods.tex` for mathematics, with independently checked notation.
- `data/ml_interpretable_v3/results_{combined,direct_only}_quick/cv_metrics.csv`: stored baselines.

Existing parent-level MAE (FeO wt.% combined/direct-only):
Gaussian Process 2.464/3.717; Extra Trees 2.723/3.226; Gradient Boosting 3.000/3.179; Random Forest 2.894/3.338; Kumar power principal 2.626/3.498 (incomplete coverage). Do NOT compare methods without matching the parent subset and reporting coverage.

## NON-GOALS for this time-limited session
- NO reading, downloading, converting, or predicting enormous real orbital M3 snapshots.
- NO geospatial maps or processing RFL/LOC/OBS imagery.
- NO deployment pipeline development or domain-screening reimplementation.
- NO large literature research, cosmetic rewrites, long mathematical PDFs, or elaborate planning.
- NO training on all Apollo data and reporting training error as cross-validation.
- NO automatic merges of new experimental branches; preserve main and open a PR if possible.

## Implement the following, in priority order

### P0 — make new code and run useful experiments immediately
1. Inspect the CSV header and target counts with a tiny Python command. Confirm dependency availability. Create `m3_feo_ml_v4/` for NEW code and `data/ml_experiments_v4/` for saved outputs. Commit incremental usable work on a new branch `claude/apollo-feo-isfeo-v4` (or another available branch if name conflicts).
2. Implement a **single configurable trainer** accepting target column, source-quality subset, feature family, model family, random seed, grouped CV folds, output dir. Keep preprocessing inside train folds. Make reproducible commands for all experiments and a small smoke test with synthetically grouped samples.
3. FeO baseline integration: read existing metrics for comparison, and perform new grouped CV experiments with at least these genuinely new choices:
   - Huber regression and Bayesian Ridge with scaling, polynomial or spline transforms of scientifically chosen compact spectral features (regularized, fold-contained).
   - HistGradientBoosting / XGBoost or LightGBM **if already installed**; else sklearn HistGradientBoosting. Try shallow trees, leaf restrictions and regularization suited to only 107 independent parents.
   - RBF kernel ridge / Gaussian process with alternative kernel and length-scale regularization if affordable; don't repeat identical prior settings.
   - Physically motivated features: visible-to-NIR spectral slope, band means/ratios and log ratios where positive, 1 μm absorption strength and continuum removal, spectral derivatives, normalized shape, selected band contrasts. Derive wavelengths from the actual band table if possible; never invent band wavelengths.
   - Modified Lucey/Kumar: fit origin and/or coefficients **only on outer training parents**, with inner grouped CV. Preserve original formula branches and coverage, and compare to fixed unmodified formulas. Avoid undefined operations and report failed coverage.
4. **Is/FeO is NOT optional.** Implement grouped-CV Is/FeO regression and at least: regularized Ridge/Huber, Extra Trees/Random Forest or HistGradientBoosting, and one nonlinear kernel model, plus a null predictor. Test raw and `log1p` targets ONLY when values are nonnegative (for strictly positive values, log is also possible). Evaluate on original target scale, ensure inverse transform is inside the estimator and folds. Inspect outliers, exact/provisional associations and counts. 29 parent groups are few; prioritize regularization and honest uncertainty over hyperparameter search.
5. Run **both** FeO settings: combined provisional `feo_wt_percent_mean` and directly-associated `feo_direct_wt_percent_mean`, and Is/FeO (if suitable data exist). Prefer 5 outer parent-grouped folds for FeO and 4–5 for Is/FeO depending on groups per fold; 3 inner group folds for reasonable tuning. Fix random seeds and save fold assignments. All per-fold preprocessing and tuning must exclude held-out parents.

### P1 — interpretation, comparison and robustness
6. Persist out-of-fold predictions with spectral row IDs, Apollo parent IDs, targets, fold, source-quality class, method, tuning parameters and prediction status; parent-mean summaries; MAE/RMSE/bias/R², train/test parent counts, missingness and coverage. Compare on identical parents; include simple parent-mean and fixed formula baselines.
7. Save final `joblib` fitted models and full reproducibility metadata under output directories; may be untracked if large but state clearly what is not pushed. Save detailed CSV outputs and plots that fit GitHub. Use relative paths.
8. Band interpretation: standardized coefficients for linear models; held-out grouped *blocks of adjacent bands* permutation importance (not just one band); optional local gradients/partial dependence for a few chosen variables. Distinguish predictive association from causal mineralogy.
9. Stress-test under grouped holdout: source-quality differences, per-parent spectral variability, scaling and small perturbations to reflectance, worst predicted outliers, negative FeO predictions and unconstrained Is/FeO predictions. Do NOT need actual M3 data.
10. If time permits, consider multi-task or residual FeO+Is/FeO modeling, but compare to simple independent Is/FeO baseline and avoid leaking true FeO into Is/FeO predictions at test time. Revisit confidence intervals by bootstrapping Apollo **parents**, not individual spectra.

## Performance & execution strategy
- First command: `git status --short && python -c "import pandas as pd; d=pd.read_csv('data/ml_combined_v2/m3_feo_isfeo_ground_truth_v2.csv'); print(d.shape); print([(c,int(d[c].notna().sum())) for c in d.columns if 'feo' in c.lower() or 'parent' in c.lower()])"`.
- Inspect code before editing. Use existing packages, sklearn and numpy; don't install GPU frameworks or massive dependencies if unnecessary.
- Do a fast smoke test, then run concise quick experiments with bounded hyperparameter grids and at most one layer of process parallelism; avoid CPU oversubscription. Save results after each model/target as it completes so the time-limited session retains value.
- Do NOT ask for permission between phases. If one method fails, log failure, keep running the others.
- Commit working code first, then complete results; explain precisely what ran and what remains. Avoid changing original ground truth.
- Every claim of numeric improvement must come from saved validation outputs and parent-grouped scoring, not apparent training fit.
- Use scoped CV to balance time: first simple fixed-hyperparameter trials, then nested tuning for promising variants. Do not spend hours on oversized searches.

## Minimum deliverable BEFORE credit expires
- Code in `m3_feo_ml_v4/` that another developer can run without this chat.
- New experiments actually executed for FeO combined, FeO strict direct-only, and Is/FeO.
- Out-of-fold predictions, parent-level metrics, comparison CSV, band-feature interpretation (at least one method), JSON run manifests and README with exact commands.
- Git commits + PR link, with a clear note if data/model binaries were not committed.
- Short final message reporting each created path, commands executed, metrics, and any errors, without claiming orbital validation.

**Start implementing now. Do not spend the remaining credit analyzing huge M3 snapshots.**
