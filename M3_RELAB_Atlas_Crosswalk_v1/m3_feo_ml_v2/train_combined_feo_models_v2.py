#!/usr/bin/env python3
"""Run parent-grouped FeO regression on the combined ground-truth table.

The outer and inner validation units are Apollo parent samples.  Results are
reported twice: by individual RELAB measurement and, more importantly, after
averaging predictions within each held-out parent so parents with many spectra
do not dominate the headline score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


VERSION = "m3-feo-regression-2.0"
BAND_COLUMNS = [f"m3_b{band:02d}" for band in range(1, 75)]
BASELINES = {
    "lucey_standard_principal": "lucey_standard_feo_principal",
    "lucey_standard_atan2_2pi": "lucey_standard_feo_atan2_2pi",
    "lucey_origin_changed_principal": "lucey_origin_changed_feo_principal",
    "lucey_origin_changed_atan2_2pi": "lucey_origin_changed_feo_atan2_2pi",
    "kumar_linear_principal": "kumar_feo_linear_principal",
    "kumar_power_principal": "kumar_feo_power_principal",
    "kumar_linear_atan2_2pi": "kumar_feo_linear_atan2_2pi",
    "kumar_power_atan2_2pi": "kumar_feo_power_atan2_2pi",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def flatten(value: object) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(-1)


def metric_values(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[valid]
    predicted = predicted[valid]
    if len(actual) == 0:
        return {
            "n": 0, "mae": math.nan, "rmse": math.nan,
            "median_absolute_error": math.nan, "bias": math.nan, "r2": math.nan,
        }
    residual = predicted - actual
    return {
        "n": int(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(actual, predicted))),
        "median_absolute_error": float(np.median(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "r2": float(r2_score(actual, predicted)) if len(actual) >= 2 else math.nan,
    }


def parent_means(actual: np.ndarray, predicted: np.ndarray,
                 groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.DataFrame({"group": groups, "actual": actual, "predicted": predicted})
    grouped = frame.groupby("group", sort=True, as_index=False).agg(
        actual=("actual", "mean"), predicted=("predicted", "mean")
    )
    return (
        grouped["actual"].to_numpy(float),
        grouped["predicted"].to_numpy(float),
        grouped["group"].astype(str).to_numpy(),
    )


def model_definitions(search_level: str, seed: int, max_pls: int):
    ridge = [0.01, 0.1, 1.0, 10.0, 100.0] if search_level == "standard" else [0.1, 1.0, 10.0]
    pls = [2, 4, 6, 8, 12, 16] if search_level == "standard" else [2, 4, 8]
    pls = [value for value in pls if value <= max_pls] or [1]
    svr = ({
        "model__C": [1.0, 10.0, 100.0],
        "model__epsilon": [0.05, 0.2, 0.5],
        "model__gamma": ["scale", 0.001, 0.01],
    } if search_level == "standard" else {
        "model__C": [1.0, 10.0],
        "model__epsilon": [0.1, 0.5],
        "model__gamma": ["scale"],
    })
    trees = ({
        "model__n_estimators": [400],
        "model__max_features": [0.35, 0.7, 1.0],
        "model__min_samples_leaf": [1, 2, 4],
    } if search_level == "standard" else {
        "model__n_estimators": [300],
        "model__max_features": [0.5, 1.0],
        "model__min_samples_leaf": [1, 3],
    })
    return {
        "ridge": (
            Pipeline([("scale", StandardScaler()), ("model", Ridge())]),
            {"model__alpha": ridge},
        ),
        "pls": (
            Pipeline([
                ("scale", StandardScaler()),
                ("model", PLSRegression(scale=False, max_iter=1000)),
            ]),
            {"model__n_components": pls},
        ),
        "svr_rbf": (
            Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="rbf"))]),
            svr,
        ),
        "extra_trees": (
            Pipeline([("model", ExtraTreesRegressor(random_state=seed, n_jobs=-1))]),
            trees,
        ),
    }


def tune(estimator, grid, x, y, groups, requested_splits: int):
    split_count = min(requested_splits, len(np.unique(groups)))
    if split_count < 2:
        first = {name: values[0] for name, values in grid.items()}
        return clone(estimator).set_params(**first).fit(x, y), first
    cv = GroupKFold(n_splits=split_count)
    grid = dict(grid)
    if "model__n_components" in grid:
        smallest = min(len(train) for train, _ in cv.split(x, y, groups))
        allowed = [value for value in grid["model__n_components"]
                   if value <= max(1, min(x.shape[1], smallest - 1))]
        grid["model__n_components"] = allowed or [1]
    search = GridSearchCV(
        clone(estimator), grid, scoring="neg_mean_absolute_error",
        cv=cv, n_jobs=-1, refit=True, error_score="raise",
    )
    search.fit(x, y, groups=groups)
    return search.best_estimator_, search.best_params_


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-column", default="feo_wt_percent_mean")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--search-level", choices=("quick", "standard"), default="quick")
    parser.add_argument("--seed", type=int, default=20261102)
    args = parser.parse_args()

    frame = pd.read_csv(args.dataset, dtype={"i_atlas_sample": str})
    required = {"id", "i_atlas_sample", args.target_column, *BAND_COLUMNS, *BASELINES.values()}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(sorted(missing))}")
    frame[args.target_column] = pd.to_numeric(frame[args.target_column], errors="coerce")
    frame = frame.loc[np.isfinite(frame[args.target_column])].copy().reset_index(drop=True)
    if frame["id"].duplicated().any():
        raise ValueError("Dataset has duplicate stable IDs")

    x = frame[BAND_COLUMNS].to_numpy(float)
    y = frame[args.target_column].to_numpy(float)
    groups = frame["i_atlas_sample"].astype(str).to_numpy()
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("M3 features and selected target must be finite")
    group_count = len(np.unique(groups))
    outer_splits = min(args.outer_folds, group_count)
    if outer_splits < 3:
        raise ValueError(f"Need at least 3 Apollo-parent groups; found {group_count}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    definitions = model_definitions(
        args.search_level, args.seed, max(1, min(len(frame) - 1, len(BAND_COLUMNS)))
    )
    predictions = {
        "training_parent_mean": np.full(len(frame), np.nan),
        **{
            name: pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
            for name, column in BASELINES.items()
        },
        **{name: np.full(len(frame), np.nan) for name in definitions},
    }

    splitter = GroupKFold(n_splits=outer_splits)
    folds = np.zeros(len(frame), dtype=int)
    fold_parameters = {name: [] for name in definitions}
    for fold, (train_index, test_index) in enumerate(splitter.split(x, y, groups), 1):
        folds[test_index] = fold
        train_frame = pd.DataFrame({"group": groups[train_index], "target": y[train_index]})
        parent_average = train_frame.groupby("group")["target"].mean().mean()
        predictions["training_parent_mean"][test_index] = float(parent_average)
        fold_defs = model_definitions(
            args.search_level, args.seed,
            max(1, min(len(train_index) - 1, x.shape[1])),
        )
        for name, (estimator, grid) in fold_defs.items():
            fitted, best = tune(
                estimator, grid, x[train_index], y[train_index], groups[train_index],
                args.inner_folds,
            )
            predictions[name][test_index] = flatten(fitted.predict(x[test_index]))
            fold_parameters[name].append({"fold": fold, "parameters": best})
        print(f"Completed parent-grouped outer fold {fold}/{outer_splits}")

    metric_rows = []
    for name, predicted in predictions.items():
        metric_rows.append({
            "method": name, "evaluation_unit": "measurement",
            **metric_values(y, predicted),
        })
        parent_y, parent_pred, _ = parent_means(y, predicted, groups)
        metric_rows.append({
            "method": name, "evaluation_unit": "apollo_parent_mean",
            **metric_values(parent_y, parent_pred),
        })
    metrics_frame = pd.DataFrame(metric_rows).sort_values(
        ["evaluation_unit", "mae", "method"]
    )
    metrics_frame.to_csv(args.output_dir / "cv_metrics.csv", index=False)

    identity = [
        "id", "i_atlas_sample", "j_child", "relab_measurement_id",
        "relab_specimen_id", "feo_target_scope", args.target_column,
        "feo_wt_percent_values", "is_feo_mean", "is_feo_values",
    ]
    identity = list(dict.fromkeys(column for column in identity if column in frame.columns))
    prediction_frame = frame[identity].copy()
    prediction_frame["grouped_cv_fold"] = folds
    for name, predicted in predictions.items():
        prediction_frame[f"prediction_{name}"] = predicted
        prediction_frame[f"residual_{name}"] = predicted - y
    prediction_frame.to_csv(args.output_dir / "cv_predictions_by_measurement.csv", index=False)

    parent_output = []
    for name, predicted in predictions.items():
        parent_y, parent_pred, parent_group = parent_means(y, predicted, groups)
        for group, actual, estimate in zip(parent_group, parent_y, parent_pred):
            parent_output.append({
                "i_atlas_sample": group, "method": name,
                "target_parent_mean": actual, "prediction_parent_mean": estimate,
                "residual": estimate - actual,
                "measurement_count": int(np.count_nonzero(groups == group)),
            })
    pd.DataFrame(parent_output).to_csv(
        args.output_dir / "cv_predictions_by_parent.csv", index=False
    )

    final_parameters = {}
    importance_rows = []
    for name, (estimator, grid) in definitions.items():
        fitted, best = tune(estimator, grid, x, y, groups, args.inner_folds)
        final_parameters[name] = best
        joblib.dump(fitted, args.output_dir / f"model_{name}.joblib")
        model = fitted.named_steps["model"]
        if name == "ridge":
            values = flatten(model.coef_)
            meaning = "standardized coefficient"
        elif name == "extra_trees":
            values = model.feature_importances_
            meaning = "impurity importance"
        else:
            continue
        for band, value in zip(BAND_COLUMNS, values):
            importance_rows.append({
                "model": name, "feature": band, "value": value, "meaning": meaning,
            })
    pd.DataFrame(importance_rows).to_csv(args.output_dir / "feature_importance.csv", index=False)
    (args.output_dir / "best_parameters.json").write_text(
        json.dumps({"final": final_parameters, "outer_folds": fold_parameters}, indent=2)
        + "\n", encoding="utf-8",
    )

    parent_metric = metrics_frame.loc[
        metrics_frame["evaluation_unit"] == "apollo_parent_mean"
    ].set_index("method")
    methods = list(parent_metric.sort_values("mae").index)
    columns = 3
    plot_rows = math.ceil(len(methods) / columns)
    figure, axes = plt.subplots(
        plot_rows, columns, figsize=(4.8 * columns, 4.4 * plot_rows), squeeze=False
    )
    all_parent_targets, _, _ = parent_means(y, y, groups)
    finite_values = [all_parent_targets]
    for name in methods:
        _, pred, _ = parent_means(y, predictions[name], groups)
        finite_values.append(pred[np.isfinite(pred)])
    values = np.concatenate([array for array in finite_values if len(array)])
    lower, upper = float(np.min(values)), float(np.max(values))
    padding = max(0.5, 0.04 * (upper - lower))
    limits = (lower - padding, upper + padding)
    for axis, name in zip(axes.flat, methods):
        actual, predicted, _ = parent_means(y, predictions[name], groups)
        valid = np.isfinite(predicted)
        axis.scatter(actual[valid], predicted[valid], s=25, alpha=0.72, edgecolors="none")
        axis.plot(limits, limits, "k--", linewidth=1)
        score = parent_metric.loc[name]
        axis.set(
            xlim=limits, ylim=limits,
            xlabel=f"Parent-mean {args.target_column}", ylabel="Predicted FeO (wt.%)",
            title=name,
        )
        axis.text(
            0.03, 0.97,
            f"MAE={score.mae:.3g}\nRMSE={score.rmse:.3g}\nR²={score.r2:.3g}",
            transform=axis.transAxes, va="top",
        )
    for axis in axes.flat[len(methods):]:
        axis.set_visible(False)
    figure.suptitle("Held-out Apollo-parent means", fontsize=14)
    figure.tight_layout()
    figure.savefig(args.output_dir / "predicted_vs_measured_parent_mean.png", dpi=180)
    plt.close(figure)

    manifest = {
        "schema": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset),
        "target_column": args.target_column,
        "rows_used": int(len(frame)),
        "apollo_parent_groups": group_count,
        "features": BAND_COLUMNS,
        "outer_folds": outer_splits,
        "inner_folds_requested": args.inner_folds,
        "search_level": args.search_level,
        "seed": args.seed,
        "primary_score": "apollo_parent_mean; every parent has equal weight",
        "reported_scores": "out-of-parent predictions only",
        "important_limitations": [
            "Many measurement rows use provisional Apollo-parent-level chemistry.",
            "RELAB geometry, grain size, maturity, and sampling differ between records.",
            "Principal-angle and atan2-mod-2pi baselines are intentionally both reported.",
            "Final joblib models are deployment fits; CV files contain honest held-out predictions.",
        ],
    }
    (args.output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print("\nApollo-parent-mean metrics (primary):")
    print(parent_metric.sort_values("mae").reset_index().to_string(index=False))
    print("\nMeasurement-level metrics:")
    print(metrics_frame.loc[metrics_frame.evaluation_unit == "measurement"]
          .sort_values("mae").to_string(index=False))
    print(f"\nRows used: {len(frame)}; Apollo parents: {group_count}")
    print(f"Results: {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
