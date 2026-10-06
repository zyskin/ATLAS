PRELIMINARY M3-BASIS FeO MACHINE LEARNING
==========================================

Purpose
-------
This experiment asks whether the 74 usable M3-equivalent RELAB reflectance
values can predict FeO.  It does not yet claim a deployable lunar map model.

The builder makes one row per unique RELAB spectral measurement.  A row is
kept only when:

  * M3 bands 1 through 74 are all usable; and
  * at least one accepted FeO value is known from RELAB chemistry, an accepted
    Atlas-PDF extraction, an adjudicated PDF/publication result, or the small
    reviewed manual-claims file.

When several distinct accepted FeO claims apply, their arithmetic mean is the
preliminary target.  The individual values and full provenance are retained in
feo_values_wt_percent and feo_claims_json.  The row ID is NOT renumbered: it is
the smallest ID already assigned to that RELAB measurement in the comprehensive
atlas_relab_feo_table.csv file.  comprehensive_table_ids records all old IDs
associated with the same spectrum.

Important scientific limitation
--------------------------------
Parent-level FeO is provisionally assigned to spectra linked to that Apollo
parent.  This can be imperfect for child fractions, separates, different grain
sizes, and heterogeneous rocks.  For that reason all spectra belonging to one
Apollo parent are held together during validation.  A random row split would
give an over-optimistic result and must not be used.

Files
-----
build_m3_feo_ml_dataset.py
    Constructs the 74-feature table and the published/legacy baseline columns.

train_m3_feo_models.py
    Runs nested, Apollo-parent-grouped cross-validation and fits final models.

manual_feo_claims_reviewed.csv
    Four manually resolved bulk values from Table 2 of the official 72435 PDF.
    Every value is sourced.  Remove --manual-feo to omit these claims.

requirements.txt
    Python packages needed for training.  The dataset builder itself uses only
    the Python standard library.

MODELS AND BASELINES
--------------------
training_parent_mean
    Predicts the mean FeO of only the training parents.  Every useful learned
    method should beat this basic reference.

lucey_direct_b09_b19
    Legacy direct calculation using M3 band 9 as R750 and band 19 as R950,
    origin (0.08, 1.19), and FeO = 17.427 * theta - 7.565.  It is included to
    reproduce the project's earlier direct-band choice.  Theta uses the
    published principal inverse-tangent form, not a wrapped 0-to-2-pi angle.
    The Clementine calibration is not expected to transfer perfectly to M3.

kumar_m3_mean_linear and kumar_m3_mean_power
    M3 adaptation described by Ajith Kumar and Kumar (2014): mean bands 8-10
    for approximately 750 nm, mean bands 18-20 for approximately 950 nm,
    origin (0.08, 1.18), followed by their linear or power FeO fit.  The paper
    preferred the power fit.

ridge
    Stable linear regression with regularisation.  A good first test when the
    74 bands are strongly correlated and the sample count is modest.

pls
    Partial least squares regression.  Commonly sensible for dense correlated
    spectra because it learns a small number of target-related spectral axes.

svr_rbf
    Smooth nonlinear regression.  It can learn curvature without requiring a
    large neural network, but is sensitive to tuning and dataset coverage.

extra_trees
    Nonlinear tree ensemble and a useful contrast to the smooth models.  It
    cannot safely extrapolate beyond the training target range.

The reported metrics are generated only from outer test folds whose Apollo
parents were absent from model fitting and hyperparameter selection.  The
saved joblib models are refitted on all rows for later use; they do not produce
the reported cross-validation scores.

LOCAL INSTALLATION
------------------
The existing gdal environment may already contain most packages, but a small
separate environment makes the run reproducible.  GDAL is NOT required for
this stage because the M3-equivalent spectra have already been produced.

  conda create -n m3-feo-ml -c conda-forge \
    python=3.12 numpy pandas scikit-learn matplotlib joblib
  conda activate m3-feo-ml

Place this directory at:

  /media/maxim/Elements/grants/2024/INSRU/pgdal/
    M3_RELAB_Atlas_Crosswalk_v1/m3_feo_ml_preliminary

BUILD THE DATASET
-----------------
Run from the Git repository root:

  cd /media/maxim/Elements/grants/2024/INSRU/pgdal

  ATLAS="$PWD/M3_RELAB_Atlas_Crosswalk_v1"
  ML="$ATLAS/m3_feo_ml_preliminary"
  OUT="$ATLAS/data/ml_preliminary_v1"

  mkdir -p "$OUT"

  python "$ML/build_m3_feo_ml_dataset.py" \
    --comprehensive-table \
      "$ATLAS/relab_raw_native_2026-10-04/atlas_relab_feo_table.csv" \
    --m3-spectra \
      "$ATLAS/data/relab_m3_v1/relab_m3_spectra.csv" \
    --atlas-pdf-feo \
      "$ATLAS/data/structured_v1/atlas_feo_measurements.csv" \
    --adjudications \
      "$ATLAS/data/ai_resolution_v1/review_candidate_adjudications.csv" \
    --missing-resolutions \
      "$ATLAS/data/ai_resolution_v1/missing_case_resolutions.csv" \
    --manual-feo \
      "$ML/manual_feo_claims_reviewed.csv" \
    --output-csv \
      "$OUT/m3_feo_ml_dataset.csv" \
    --manifest-json \
      "$OUT/m3_feo_ml_dataset_manifest.json"

If either AI-resolution CSV has a different directory name, locate it first:

  find "$ATLAS/data" -type f \
    \( -name review_candidate_adjudications.csv \
       -o -name missing_case_resolutions.csv \) -print

Then replace only the corresponding path in the command.  Optional source
files may also be omitted completely; the manifest records exactly which
inputs were used and their SHA-256 hashes.

Inspect the resulting sample size before fitting:

  python - "$OUT/m3_feo_ml_dataset.csv" <<'PY'
  import pandas as pd, sys
  d = pd.read_csv(sys.argv[1], dtype={"i_atlas_sample": str})
  print("Spectra:", len(d))
  print("Apollo parents:", d.i_atlas_sample.nunique())
  print("FeO range:", d.feo_mean_wt_percent.min(), "to",
        d.feo_mean_wt_percent.max(), "wt.%")
  print(d.feo_claim_count.value_counts().sort_index())
  PY

TRAIN AND COMPARE
-----------------
Quick preliminary run:

  time python "$ML/train_m3_feo_models.py" \
    --dataset "$OUT/m3_feo_ml_dataset.csv" \
    --output-dir "$OUT/results_quick" \
    --search-level quick

Broader, slower hyperparameter search after the quick run succeeds:

  time python "$ML/train_m3_feo_models.py" \
    --dataset "$OUT/m3_feo_ml_dataset.csv" \
    --output-dir "$OUT/results_standard" \
    --search-level standard

Main outputs
------------
m3_feo_ml_dataset.csv
    Modeling table with stable IDs, FeO claim provenance, baseline calculations,
    and m3_b01 through m3_b74.

cv_metrics.csv
    MAE, RMSE, median absolute error, bias, and R-squared.  Rank primarily by
    MAE/RMSE; negative R-squared means worse than a constant prediction.

cv_predictions.csv
    Out-of-parent prediction and residual for every stable ID and method.

predicted_vs_measured.png
    One measured-versus-predicted panel per baseline/model.

best_parameters.json
    Selected settings for each fold and the final all-data fit.

model_*.joblib
    Final models fitted to the full preliminary table.  Preserve the matching
    training_manifest.json and feature order when using them.

feature_importance.csv
    Ridge standardized coefficients and Extra Trees impurity importances.  They
    are diagnostic only and are not direct measures of mineral physics.

PUSH ONLY THIS WORK TO GITHUB
-----------------------------
From /media/maxim/Elements/grants/2024/INSRU/pgdal:

  git status --short

  git add -- \
    M3_RELAB_Atlas_Crosswalk_v1/m3_feo_ml_preliminary \
    M3_RELAB_Atlas_Crosswalk_v1/data/ml_preliminary_v1

  git diff --cached --stat
  git status --short

Check that only the two paths above are staged.  Then:

  git commit -m "Add preliminary M3 FeO regression experiment"
  git push origin main

If joblib files are unexpectedly large, do not add them.  Unstage only those:

  git restore --staged \
    M3_RELAB_Atlas_Crosswalk_v1/data/ml_preliminary_v1/results_quick/model_*.joblib \
    M3_RELAB_Atlas_Crosswalk_v1/data/ml_preliminary_v1/results_standard/model_*.joblib

The CSV results, plots, parameters, and manifests are sufficient for later AI
analysis even if the fitted model binaries are not pushed.

Reference used for the M3 averaging baseline
--------------------------------------------
Ajith Kumar, S. and Kumar, A. S. K. (2014), “Estimation of iron abundance from
Moon Mineralogy Mapper data: An assessment of the uncertainty of the method,”
Advances in Space Research 53, 698-710. DOI: 10.1016/j.asr.2014.01.009.
