#!/usr/bin/env python3
"""Fit, compare, preserve, and interpret FeO regression models.

Every validation split is made by Apollo parent sample.  Hyperparameters are
selected by an inner parent-grouped cross-validation whose objective is the
mean absolute error after averaging predictions within each validation parent.
The script saves honest outer-fold predictions, final deployment fits, explicit
model parameters where they exist, held-out permutation importance, and a
training-domain model used to screen real M3 spectra before interpretation.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


VERSION = "m3-feo-regression-interpretability-3.0"
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


def flatten(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def json_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return repr(value)


def metric_values(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[valid]
    predicted = predicted[valid]
    if len(actual) == 0:
        return {
            "n": 0,
            "mae": math.nan,
            "rmse": math.nan,
            "median_absolute_error": math.nan,
            "bias": math.nan,
            "r2": math.nan,
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


def parent_means(
    actual: np.ndarray, predicted: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = pd.DataFrame({"group": groups, "actual": actual, "predicted": predicted})
    table = table.loc[np.isfinite(table["actual"]) & np.isfinite(table["predicted"])]
    grouped = table.groupby("group", sort=True, as_index=False).agg(
        actual=("actual", "mean"), predicted=("predicted", "mean")
    )
    return (
        grouped["actual"].to_numpy(float),
        grouped["predicted"].to_numpy(float),
        grouped["group"].astype(str).to_numpy(),
    )


def parent_mae(actual: np.ndarray, predicted: np.ndarray, groups: np.ndarray) -> float:
    group_actual, group_predicted, _ = parent_means(actual, predicted, groups)
    if len(group_actual) == 0:
        return math.inf
    return float(mean_absolute_error(group_actual, group_predicted))


def model_definitions(search_level: str, seed: int, maximum_components: int):
    if search_level == "standard":
        ridge_alpha = [0.01, 0.1, 1.0, 10.0, 100.0]
        component_counts = [2, 4, 6, 8, 12, 16, 24]
        elastic_alpha = [0.001, 0.01, 0.1, 1.0]
        elastic_ratio = [0.05, 0.25, 0.5, 0.75, 0.95]
    else:
        ridge_alpha = [0.1, 1.0, 10.0]
        component_counts = [2, 4, 8, 16]
        elastic_alpha = [0.01, 0.1, 1.0]
        elastic_ratio = [0.1, 0.5, 0.9]
    component_counts = [v for v in component_counts if v <= maximum_components] or [1]
    pcr_counts = [v for v in [5, 10, 20, 35] if v <= maximum_components] or [1]
    gpr_counts = [v for v in [5, 10, 20] if v <= maximum_components] or [1]

    fixed_kernel = (
        ConstantKernel(1.0, constant_value_bounds="fixed")
        * RBF(length_scale=3.0, length_scale_bounds="fixed")
        + WhiteKernel(noise_level=0.1, noise_level_bounds="fixed")
    )
    definitions = {
        "ridge": {
            "estimator": Pipeline([
                ("scale", StandardScaler()),
                ("model", Ridge()),
            ]),
            "grid": {"model__alpha": ridge_alpha},
            "family": "linear_regularized",
        },
        "elastic_net": {
            "estimator": Pipeline([
                ("scale", StandardScaler()),
                ("model", ElasticNet(max_iter=100000, random_state=seed)),
            ]),
            "grid": {
                "model__alpha": elastic_alpha,
                "model__l1_ratio": elastic_ratio,
            },
            "family": "linear_regularized_sparse",
        },
        "pcr_ridge": {
            "estimator": Pipeline([
                ("scale", StandardScaler()),
                ("pca", PCA(random_state=seed)),
                ("model", Ridge()),
            ]),
            "grid": {
                "pca__n_components": pcr_counts,
                "model__alpha": ridge_alpha,
            },
            "family": "principal_component_regression",
        },
        "pls": {
            "estimator": Pipeline([
                ("scale", StandardScaler()),
                ("model", PLSRegression(scale=False, max_iter=2000)),
            ]),
            "grid": {"model__n_components": component_counts},
            "family": "partial_least_squares",
        },
        "svr_rbf": {
            "estimator": Pipeline([
                ("scale", StandardScaler()),
                ("model", SVR(kernel="rbf")),
            ]),
            "grid": ({
                "model__C": [1.0, 10.0, 100.0],
                "model__epsilon": [0.05, 0.2, 0.5],
                "model__gamma": ["scale", 0.001, 0.01, 0.1],
            } if search_level == "standard" else {
                "model__C": [1.0, 10.0],
                "model__epsilon": [0.1, 0.5],
                "model__gamma": ["scale", 0.01],
            }),
            "family": "kernel_regression",
        },
        "extra_trees": {
            "estimator": Pipeline([
                ("model", ExtraTreesRegressor(random_state=seed, n_jobs=-1)),
            ]),
            "grid": ({
                "model__n_estimators": [600],
                "model__max_features": [0.25, 0.5, 0.75, 1.0],
                "model__min_samples_leaf": [1, 2, 4, 8],
            } if search_level == "standard" else {
                "model__n_estimators": [400],
                "model__max_features": [0.5, 1.0],
                "model__min_samples_leaf": [1, 3],
            }),
            "family": "randomized_tree_ensemble",
        },
        "random_forest": {
            "estimator": Pipeline([
                ("model", RandomForestRegressor(
                    random_state=seed, n_jobs=-1, bootstrap=True
                )),
            ]),
            "grid": ({
                "model__n_estimators": [600],
                "model__max_features": [0.25, 0.5, 0.75, 1.0],
                "model__min_samples_leaf": [1, 2, 4, 8],
            } if search_level == "standard" else {
                "model__n_estimators": [400],
                "model__max_features": [0.5, 1.0],
                "model__min_samples_leaf": [1, 3],
            }),
            "family": "bootstrap_tree_ensemble",
        },
        "gradient_boosting": {
            "estimator": Pipeline([
                ("model", GradientBoostingRegressor(
                    random_state=seed, loss="huber"
                )),
            ]),
            "grid": ({
                "model__n_estimators": [100, 250, 500],
                "model__learning_rate": [0.02, 0.05, 0.1],
                "model__max_depth": [1, 2, 3],
                "model__min_samples_leaf": [2, 5, 10],
            } if search_level == "standard" else {
                "model__n_estimators": [100, 300],
                "model__learning_rate": [0.03, 0.1],
                "model__max_depth": [1, 2],
                "model__min_samples_leaf": [3],
            }),
            "family": "sequential_tree_ensemble",
        },
        "gaussian_process": {
            "estimator": Pipeline([
                ("scale", StandardScaler()),
                ("pca", PCA(whiten=True, random_state=seed)),
                ("model", GaussianProcessRegressor(
                    kernel=fixed_kernel,
                    optimizer=None,
                    normalize_y=True,
                    random_state=seed,
                )),
            ]),
            "grid": {
                "pca__n_components": gpr_counts,
                "model__alpha": ([1e-6, 1e-4, 1e-2, 0.1]
                                  if search_level == "standard"
                                  else [1e-4, 1e-2]),
            },
            "family": "probabilistic_kernel_regression",
        },
    }
    return definitions


def valid_parameter_grid(
    grid: dict[str, list[Any]], estimator: Pipeline, x_rows: int, x_columns: int
) -> list[dict[str, Any]]:
    candidates = list(ParameterGrid(grid))
    allowed = []
    for candidate in candidates:
        component_values = [
            value for key, value in candidate.items()
            if key.endswith("__n_components")
        ]
        if component_values and max(component_values) > min(x_columns, x_rows - 1):
            continue
        allowed.append(candidate)
    return allowed or [{}]


def tune_by_parent_mae(
    estimator: Pipeline,
    grid: dict[str, list[Any]],
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    requested_splits: int,
) -> tuple[Pipeline, dict[str, Any], list[dict[str, Any]]]:
    split_count = min(requested_splits, len(np.unique(groups)))
    candidates = valid_parameter_grid(grid, estimator, len(x), x.shape[1])
    if split_count < 2:
        fitted = clone(estimator).set_params(**candidates[0]).fit(x, y)
        return fitted, candidates[0], []

    splitter = GroupKFold(n_splits=split_count)
    split_indices = list(splitter.split(x, y, groups))
    results = []
    best_score = math.inf
    best_parameters: dict[str, Any] | None = None
    for candidate_number, parameters in enumerate(candidates, 1):
        fold_scores = []
        failed = None
        for train_index, validation_index in split_indices:
            try:
                fitted = clone(estimator).set_params(**parameters)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fitted.fit(x[train_index], y[train_index])
                prediction = flatten(fitted.predict(x[validation_index]))
                fold_scores.append(parent_mae(
                    y[validation_index], prediction, groups[validation_index]
                ))
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                failed = f"{type(exc).__name__}: {exc}"
                fold_scores = [math.inf]
                break
        mean_score = float(np.mean(fold_scores))
        results.append({
            "candidate": candidate_number,
            "parameters": json_value(parameters),
            "inner_parent_mae": mean_score,
            "fold_parent_mae": fold_scores,
            "failure": failed,
        })
        if mean_score < best_score:
            best_score = mean_score
            best_parameters = parameters
    if best_parameters is None:
        raise RuntimeError("Every hyperparameter candidate failed")
    fitted = clone(estimator).set_params(**best_parameters)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted.fit(x, y)
    return fitted, best_parameters, results


def effective_linear_coefficients(
    name: str, fitted: Pipeline
) -> tuple[np.ndarray, np.ndarray, float] | None:
    if "scale" not in fitted.named_steps:
        return None
    scaler = fitted.named_steps["scale"]
    model = fitted.named_steps["model"]
    if name in {"ridge", "elastic_net"}:
        standardized = flatten(model.coef_)
        standardized_intercept = float(flatten([model.intercept_])[0])
    elif name == "pcr_ridge":
        pca = fitted.named_steps["pca"]
        standardized = pca.components_.T @ flatten(model.coef_)
        standardized_intercept = float(flatten([model.intercept_])[0])
        standardized_intercept -= float(pca.mean_ @ standardized)
    elif name == "pls":
        standardized = flatten(model.coef_)
        standardized_intercept = float(flatten(model.intercept_)[0])
    else:
        return None
    original = standardized / scaler.scale_
    original_intercept = standardized_intercept - float(np.sum(
        standardized * scaler.mean_ / scaler.scale_
    ))
    return standardized, original, original_intercept


def save_component_matrices(name: str, fitted: Pipeline, output_dir: Path) -> None:
    if name == "pcr_ridge":
        pca = fitted.named_steps["pca"]
        rows = []
        for component, coefficients in enumerate(pca.components_, 1):
            for band, coefficient in zip(BAND_COLUMNS, coefficients):
                rows.append({
                    "component": component,
                    "band": band,
                    "basis_coefficient": coefficient,
                    "explained_variance": pca.explained_variance_[component - 1],
                    "explained_variance_ratio": pca.explained_variance_ratio_[component - 1],
                })
        pd.DataFrame(rows).to_csv(output_dir / "pcr_components.csv", index=False)
    elif name == "pls":
        model = fitted.named_steps["model"]
        rows = []
        for component in range(model.x_weights_.shape[1]):
            for band_index, band in enumerate(BAND_COLUMNS):
                rows.append({
                    "component": component + 1,
                    "band": band,
                    "x_weight": model.x_weights_[band_index, component],
                    "x_coefficient_vector": model.x_loadings_[band_index, component],
                    "x_rotation": model.x_rotations_[band_index, component],
                    "y_weight": flatten(model.y_weights_[:, component])[0],
                    "y_coefficient_vector": flatten(model.y_loadings_[:, component])[0],
                })
        pd.DataFrame(rows).to_csv(output_dir / "pls_components.csv", index=False)


def save_svr_details(
    fitted: Pipeline, frame: pd.DataFrame, x: np.ndarray, output_dir: Path
) -> pd.DataFrame:
    scaler = fitted.named_steps["scale"]
    model = fitted.named_steps["model"]
    support_indices = model.support_.astype(int)
    identity = [c for c in [
        "id", "i_atlas_sample", "j_child", "relab_measurement_id",
        "relab_specimen_id",
    ] if c in frame.columns]
    support = frame.iloc[support_indices][identity].copy().reset_index(drop=True)
    support.insert(0, "support_vector_number", np.arange(1, len(support) + 1))
    support["dual_coefficient"] = flatten(model.dual_coef_)
    support["intercept"] = float(flatten(model.intercept_)[0])
    support["gamma"] = float(model._gamma)
    support.to_csv(output_dir / "svr_support_vectors.csv", index=False)

    z = scaler.transform(x)
    sv = model.support_vectors_
    dual = flatten(model.dual_coef_)
    gamma = float(model._gamma)
    sample_count = len(z)
    signed_sum = np.zeros(z.shape[1])
    absolute_sum = np.zeros(z.shape[1])
    batch_size = 128
    for start in range(0, sample_count, batch_size):
        current = z[start:start + batch_size]
        squared = (
            np.sum(current * current, axis=1)[:, None]
            + np.sum(sv * sv, axis=1)[None, :]
            - 2.0 * current @ sv.T
        )
        kernel = np.exp(-gamma * np.maximum(squared, 0.0))
        gradient = 2.0 * gamma * (
            (kernel * dual[None, :]) @ sv
            - np.sum(kernel * dual[None, :], axis=1)[:, None] * current
        )
        signed_sum += np.sum(gradient, axis=0)
        absolute_sum += np.sum(np.abs(gradient), axis=0)
    sensitivity = pd.DataFrame({
        "model": "svr_rbf",
        "feature": BAND_COLUMNS,
        "mean_signed_derivative_standardized": signed_sum / sample_count,
        "mean_absolute_derivative_standardized": absolute_sum / sample_count,
    })
    sensitivity.to_csv(output_dir / "svr_band_sensitivity.csv", index=False)
    return sensitivity


def build_domain_reference(
    x: np.ndarray,
    y: np.ndarray,
    frame: pd.DataFrame,
    dataset_hash: str,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    scaler = StandardScaler().fit(x)
    standardized = scaler.transform(x)
    pca = PCA(n_components=0.995, svd_solver="full", random_state=seed).fit(standardized)
    scores = pca.transform(standardized)
    score_mean = np.mean(scores, axis=0)
    score_variance = np.var(scores, axis=0, ddof=1)
    variance_floor = max(float(np.max(score_variance)) * 1e-10, 1e-12)
    distance_squared = np.sum(
        (scores - score_mean) ** 2 / np.maximum(score_variance, variance_floor),
        axis=1,
    )
    thresholds = {
        "p95": float(np.quantile(distance_squared, 0.95)),
        "p99": float(np.quantile(distance_squared, 0.99)),
        "maximum": float(np.max(distance_squared)),
    }
    reference = {
        "schema": "m3-feo-training-domain-1.0",
        "band_columns": BAND_COLUMNS,
        "scaler": scaler,
        "pca": pca,
        "pca_score_mean": score_mean,
        "pca_score_variance": score_variance,
        "pca_variance_floor": variance_floor,
        "pca_distance_squared_thresholds": thresholds,
        "band_minimum": np.min(x, axis=0),
        "band_maximum": np.max(x, axis=0),
        "band_q01": np.quantile(x, 0.01, axis=0),
        "band_q99": np.quantile(x, 0.99, axis=0),
        "target_minimum": float(np.min(y)),
        "target_maximum": float(np.max(y)),
        "dataset_sha256": dataset_hash,
    }
    joblib.dump(reference, output_dir / "training_domain_reference.joblib")
    statistics = pd.DataFrame({
        "band": BAND_COLUMNS,
        "minimum": reference["band_minimum"],
        "q01": reference["band_q01"],
        "mean": np.mean(x, axis=0),
        "standard_deviation": np.std(x, axis=0, ddof=1),
        "q99": reference["band_q99"],
        "maximum": reference["band_maximum"],
    })
    statistics.to_csv(output_dir / "training_band_statistics.csv", index=False)
    identity = [c for c in [
        "id", "i_atlas_sample", "j_child", "relab_measurement_id",
    ] if c in frame.columns]
    reference_table = pd.concat(
        [frame[identity].reset_index(drop=True), frame[BAND_COLUMNS].reset_index(drop=True)],
        axis=1,
    )
    reference_table.to_csv(output_dir / "training_reference_spectra.csv", index=False)
    return {
        "pca_component_count": int(pca.n_components_),
        "pca_explained_variance_fraction": float(np.sum(pca.explained_variance_ratio_)),
        "pca_distance_squared_thresholds": thresholds,
        "target_minimum": float(np.min(y)),
        "target_maximum": float(np.max(y)),
    }


def plot_comparison(
    metrics: pd.DataFrame, predictions: dict[str, np.ndarray], y: np.ndarray,
    groups: np.ndarray, output_dir: Path, target_name: str
) -> None:
    parent_metric = metrics.loc[
        metrics["evaluation_unit"] == "apollo_parent_mean"
    ].set_index("method")
    methods = list(parent_metric.sort_values("mae").index)
    columns = 3
    rows = math.ceil(len(methods) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4.8 * columns, 4.4 * rows), squeeze=False)
    all_targets, _, _ = parent_means(y, y, groups)
    values = [all_targets]
    for name in methods:
        _, p, _ = parent_means(y, predictions[name], groups)
        values.append(p[np.isfinite(p)])
    combined = np.concatenate([v for v in values if len(v)])
    lower, upper = float(np.min(combined)), float(np.max(combined))
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
            xlabel=f"Parent-mean {target_name}", ylabel="Predicted FeO (wt.%)",
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
    figure.savefig(output_dir / "predicted_vs_measured_parent_mean.png", dpi=180)
    plt.close(figure)


def plot_interpretation(
    permutation_table: pd.DataFrame,
    coefficient_table: pd.DataFrame,
    tree_table: pd.DataFrame,
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    if not permutation_table.empty:
        for model, subset in permutation_table.groupby("model"):
            axes[0].plot(
                np.arange(1, 75), subset.sort_values("band_number")["mae_increase_mean"],
                label=model, linewidth=1.1,
            )
        axes[0].set_ylabel("Held-out MAE increase")
        axes[0].set_title("Permutation importance; individual correlated bands")
        axes[0].legend(ncol=3, fontsize=7)
    if not coefficient_table.empty:
        for model, subset in coefficient_table.groupby("model"):
            axes[1].plot(
                np.arange(1, 75), subset.sort_values("band_number")["coefficient_standardized"],
                label=model, linewidth=1.2,
            )
        axes[1].axhline(0.0, color="black", linewidth=0.7)
        axes[1].set_ylabel("Coefficient per 1 SD")
        axes[1].set_title("Effective linear band coefficients")
        axes[1].legend(ncol=4, fontsize=7)
    if not tree_table.empty:
        for model, subset in tree_table.groupby("model"):
            axes[2].plot(
                np.arange(1, 75), subset.sort_values("band_number")["impurity_importance"],
                label=model, linewidth=1.2,
            )
        axes[2].set_ylabel("Impurity reduction fraction")
        axes[2].set_title("Tree-ensemble split importance")
        axes[2].legend(ncol=3, fontsize=8)
    axes[2].set_xlabel("M3 band number")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "model_band_interpretation.png", dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-column", default="feo_wt_percent_mean")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--search-level", choices=("quick", "standard"), default="quick")
    parser.add_argument("--permutation-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261103)
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
    if args.permutation_repeats < 1:
        raise ValueError("--permutation-repeats must be at least 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    maximum_components = max(1, min(len(frame) - 1, x.shape[1]))
    definitions = model_definitions(args.search_level, args.seed, maximum_components)
    predictions: dict[str, np.ndarray] = {
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
    permutation_rows = []
    for fold, (train_index, test_index) in enumerate(splitter.split(x, y, groups), 1):
        folds[test_index] = fold
        train_table = pd.DataFrame({"group": groups[train_index], "target": y[train_index]})
        parent_average = train_table.groupby("group")["target"].mean().mean()
        predictions["training_parent_mean"][test_index] = float(parent_average)
        fold_definitions = model_definitions(
            args.search_level, args.seed + fold,
            max(1, min(len(train_index) - 1, x.shape[1])),
        )
        for name, definition in fold_definitions.items():
            fitted, best, inner_results = tune_by_parent_mae(
                definition["estimator"], definition["grid"],
                x[train_index], y[train_index], groups[train_index], args.inner_folds,
            )
            predictions[name][test_index] = flatten(fitted.predict(x[test_index]))
            fold_parameters[name].append({
                "fold": fold,
                "parameters": json_value(best),
                "inner_search": inner_results,
            })
            importance = permutation_importance(
                fitted, x[test_index], y[test_index],
                scoring="neg_mean_absolute_error",
                n_repeats=args.permutation_repeats,
                random_state=args.seed + 1000 * fold,
                n_jobs=-1,
            )
            for band_number, band in enumerate(BAND_COLUMNS, 1):
                for repeat in range(args.permutation_repeats):
                    permutation_rows.append({
                        "model": name,
                        "outer_fold": fold,
                        "band": band,
                        "band_number": band_number,
                        "repeat": repeat + 1,
                        "measurement_mae_increase": importance.importances[band_number - 1, repeat],
                    })
        print(f"Completed parent-grouped outer fold {fold}/{outer_splits}", flush=True)

    metric_rows = []
    for name, predicted in predictions.items():
        metric_rows.append({
            "method": name,
            "evaluation_unit": "measurement",
            **metric_values(y, predicted),
        })
        parent_y, parent_predicted, _ = parent_means(y, predicted, groups)
        metric_rows.append({
            "method": name,
            "evaluation_unit": "apollo_parent_mean",
            **metric_values(parent_y, parent_predicted),
        })
    metrics = pd.DataFrame(metric_rows).sort_values(["evaluation_unit", "mae", "method"])
    metrics.to_csv(args.output_dir / "cv_metrics.csv", index=False)

    identity = [
        "id", "i_atlas_sample", "j_child", "relab_measurement_id",
        "relab_specimen_id", "feo_target_scope", args.target_column,
        "feo_wt_percent_values", "is_feo_mean", "is_feo_values",
    ]
    identity = list(dict.fromkeys(column for column in identity if column in frame.columns))
    measurement_output = frame[identity].copy()
    measurement_output["grouped_cv_fold"] = folds
    for name, predicted in predictions.items():
        measurement_output[f"prediction_{name}"] = predicted
        measurement_output[f"residual_{name}"] = predicted - y
    measurement_output.to_csv(args.output_dir / "cv_predictions_by_measurement.csv", index=False)

    parent_output = []
    for name, predicted in predictions.items():
        parent_y, parent_predicted, parent_group = parent_means(y, predicted, groups)
        for group, actual, estimate in zip(parent_group, parent_y, parent_predicted):
            parent_output.append({
                "i_atlas_sample": group,
                "method": name,
                "target_parent_mean": actual,
                "prediction_parent_mean": estimate,
                "residual": estimate - actual,
                "measurement_count": int(np.count_nonzero(groups == group)),
            })
    pd.DataFrame(parent_output).to_csv(args.output_dir / "cv_predictions_by_parent.csv", index=False)

    raw_permutation = pd.DataFrame(permutation_rows)
    raw_permutation.to_csv(args.output_dir / "cv_permutation_importance_repeats.csv", index=False)
    permutation_summary = raw_permutation.groupby(
        ["model", "band", "band_number"], as_index=False
    ).agg(
        mae_increase_mean=("measurement_mae_increase", "mean"),
        mae_increase_standard_deviation=("measurement_mae_increase", "std"),
        values=("measurement_mae_increase", "count"),
    )
    permutation_summary.to_csv(args.output_dir / "cv_permutation_importance.csv", index=False)

    final_parameters = {}
    final_models = {}
    coefficient_rows = []
    tree_rows = []
    for name, definition in definitions.items():
        fitted, best, search_results = tune_by_parent_mae(
            definition["estimator"], definition["grid"], x, y, groups, args.inner_folds
        )
        final_models[name] = fitted
        final_parameters[name] = {
            "family": definition["family"],
            "parameters": json_value(best),
            "inner_search": search_results,
        }
        joblib.dump(fitted, args.output_dir / f"model_{name}.joblib")
        effective = effective_linear_coefficients(name, fitted)
        if effective is not None:
            standardized, original, original_intercept = effective
            for band_number, (band, c_standard, c_original) in enumerate(
                zip(BAND_COLUMNS, standardized, original), 1
            ):
                coefficient_rows.append({
                    "model": name,
                    "band": band,
                    "band_number": band_number,
                    "coefficient_standardized": c_standard,
                    "coefficient_original_reflectance_units": c_original,
                    "intercept_original_reflectance_units": original_intercept,
                })
        model = fitted.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            for band_number, (band, value) in enumerate(
                zip(BAND_COLUMNS, model.feature_importances_), 1
            ):
                tree_rows.append({
                    "model": name,
                    "band": band,
                    "band_number": band_number,
                    "impurity_importance": value,
                })
        save_component_matrices(name, fitted, args.output_dir)

    coefficient_table = pd.DataFrame(coefficient_rows)
    tree_table = pd.DataFrame(tree_rows)
    coefficient_table.to_csv(args.output_dir / "linear_band_coefficients.csv", index=False)
    tree_table.to_csv(args.output_dir / "tree_feature_importance.csv", index=False)
    svr_sensitivity = save_svr_details(
        final_models["svr_rbf"], frame, x, args.output_dir
    )

    # Put the different, explicitly non-equivalent interpretation measures in
    # one ranked table.  Keeping ``measure`` in the table prevents an impurity
    # importance, a regression coefficient, and a held-out permutation effect
    # from being mistaken for the same mathematical quantity.
    ranked_rows = []
    for model_name, table in permutation_summary.groupby("model"):
        ordered = table.sort_values("mae_increase_mean", ascending=False)
        for rank, row in enumerate(ordered.itertuples(index=False), 1):
            ranked_rows.append({
                "model": model_name,
                "measure": "held_out_permutation_mae_increase",
                "rank_within_model_and_measure": rank,
                "band": row.band,
                "band_number": row.band_number,
                "value": row.mae_increase_mean,
                "signed_value": row.mae_increase_mean,
            })
    for model_name, table in coefficient_table.groupby("model"):
        ordered = table.assign(
            magnitude=table["coefficient_standardized"].abs()
        ).sort_values("magnitude", ascending=False)
        for rank, row in enumerate(ordered.itertuples(index=False), 1):
            ranked_rows.append({
                "model": model_name,
                "measure": "absolute_standardized_linear_coefficient",
                "rank_within_model_and_measure": rank,
                "band": row.band,
                "band_number": row.band_number,
                "value": abs(row.coefficient_standardized),
                "signed_value": row.coefficient_standardized,
            })
    for model_name, table in tree_table.groupby("model"):
        ordered = table.sort_values("impurity_importance", ascending=False)
        for rank, row in enumerate(ordered.itertuples(index=False), 1):
            ranked_rows.append({
                "model": model_name,
                "measure": "tree_impurity_importance",
                "rank_within_model_and_measure": rank,
                "band": row.band,
                "band_number": row.band_number,
                "value": row.impurity_importance,
                "signed_value": np.nan,
            })
    ordered = svr_sensitivity.sort_values(
        "mean_absolute_derivative_standardized", ascending=False
    )
    for rank, row in enumerate(ordered.itertuples(index=False), 1):
        ranked_rows.append({
            "model": "svr_rbf",
            "measure": "mean_absolute_local_derivative_standardized",
            "rank_within_model_and_measure": rank,
            "band": row.feature,
            "band_number": int(str(row.feature).replace("m3_b", "")),
            "value": row.mean_absolute_derivative_standardized,
            "signed_value": row.mean_signed_derivative_standardized,
        })
    pd.DataFrame(ranked_rows).to_csv(
        args.output_dir / "band_interpretation_ranked.csv", index=False
    )
    (args.output_dir / "best_parameters.json").write_text(
        json.dumps({"final": final_parameters, "outer_folds": fold_parameters}, indent=2)
        + "\n",
        encoding="utf-8",
    )

    dataset_hash = sha256(args.dataset)
    domain_summary = build_domain_reference(
        x, y, frame, dataset_hash, args.output_dir, args.seed
    )
    deployment_manifest = {
        "schema": "m3-feo-deployment-bundle-1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_names": list(definitions),
        "model_files": {name: f"model_{name}.joblib" for name in definitions},
        "domain_reference_file": "training_domain_reference.joblib",
        "band_columns": BAND_COLUMNS,
        "target_column": args.target_column,
        "dataset_sha256": dataset_hash,
        "domain_summary": domain_summary,
        "prediction_policy": {
            "raw_predictions_are_never_clipped": True,
            "screen_before_interpretation": True,
            "recommended_physical_feo_interval_wt_percent": [0.0, 30.0],
            "do_not_treat_tree_variation_or_model_disagreement_as_calibrated_uncertainty": True,
        },
    }
    (args.output_dir / "deployment_manifest.json").write_text(
        json.dumps(deployment_manifest, indent=2) + "\n", encoding="utf-8"
    )

    training_predictions = frame[[c for c in [
        "id", "i_atlas_sample", "j_child", "relab_measurement_id", args.target_column
    ] if c in frame.columns]].copy()
    for name, fitted in final_models.items():
        training_predictions[f"fitted_{name}"] = flatten(fitted.predict(x))
    training_predictions.to_csv(args.output_dir / "final_fit_training_predictions.csv", index=False)

    plot_comparison(metrics, predictions, y, groups, args.output_dir, args.target_column)
    plot_interpretation(
        permutation_summary, coefficient_table, tree_table, args.output_dir
    )

    manifest = {
        "schema": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": dataset_hash,
        "target_column": args.target_column,
        "rows_used": int(len(frame)),
        "apollo_parent_groups": group_count,
        "features": BAND_COLUMNS,
        "models_fitted": list(definitions),
        "empirical_baselines": list(BASELINES),
        "outer_folds": outer_splits,
        "inner_folds_requested": args.inner_folds,
        "hyperparameter_objective": "mean absolute error of validation-parent mean predictions",
        "permutation_repeats": args.permutation_repeats,
        "search_level": args.search_level,
        "seed": args.seed,
        "primary_score": "apollo_parent_mean; every held-out parent has equal evaluation weight",
        "important_limitations": [
            "Many rows use provisional Apollo-parent chemistry assigned to child or fraction spectra.",
            "Repeated spectra from a parent are not statistically independent.",
            "Training fits are preserved for deployment; outer-fold predictions are the performance evidence.",
            "Permutation of one band is difficult to interpret because adjacent M3 bands are correlated.",
            "No fitted model is permitted to bypass the orbital-spectrum domain diagnostics.",
        ],
    }
    (args.output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    parent_metric = metrics.loc[
        metrics["evaluation_unit"] == "apollo_parent_mean"
    ].sort_values("mae")
    print("\nApollo-parent-mean metrics (primary):")
    print(parent_metric.to_string(index=False))
    print(f"\nRows used: {len(frame)}; Apollo parents: {group_count}")
    print(f"Results: {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
