"""Model families and a self-contained modified Lucey/Kumar estimator for v4.

Each model family is a factory returning ``(estimator, param_grid)`` where
``estimator`` is an sklearn Pipeline whose *first* step is a ``StandardScaler``
(so all scaling is fitted inside the training fold) and ``param_grid`` is a
small, regularisation-focused grid suited to ~30-110 independent Apollo parents.

The ``SpectralAngleRegressor`` reimplements the Lucey/Kumar ferrous-angle model
so its origin and coefficients are fitted *only on training parents* (never the
published constants), enabling an honest "modified formula" comparison against
the fixed published columns already stored in the dataset.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, HuberRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler


# --------------------------------------------------------------------------- #
# Linear / kernel / tree model families (feature matrix supplied by trainer).  #
# --------------------------------------------------------------------------- #
def _huber(seed: int):
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", HuberRegressor(max_iter=6000)),
        ]
    )
    grid = {
        "model__alpha": [1e-3, 1e-2, 1e-1, 1.0],
        "model__epsilon": [1.2, 1.35, 1.6],
    }
    return pipe, grid


def _bayesian_ridge(seed: int):
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", BayesianRidge()),
        ]
    )
    grid = {
        "model__alpha_1": [1e-6, 1e-4],
        "model__lambda_1": [1e-6, 1e-4],
    }
    return pipe, grid


def _huber_spline(seed: int):
    """Huber on monotone spline transforms of compact (physical) features."""
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("spline", SplineTransformer(degree=3, n_knots=4, include_bias=False)),
            ("model", HuberRegressor(max_iter=6000)),
        ]
    )
    grid = {
        "spline__n_knots": [3, 4],
        "model__alpha": [1e-2, 1e-1, 1.0],
    }
    return pipe, grid


def _bayesian_ridge_spline(seed: int):
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("spline", SplineTransformer(degree=3, n_knots=4, include_bias=False)),
            ("model", BayesianRidge()),
        ]
    )
    grid = {"spline__n_knots": [3, 4, 5]}
    return pipe, grid


def _ridge(seed: int):
    pipe = Pipeline([("scale", StandardScaler()), ("model", Ridge())])
    grid = {"model__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]}
    return pipe, grid


def _hist_gbm(seed: int):
    pipe = Pipeline(
        [
            ("scale", StandardScaler(with_mean=True, with_std=True)),
            (
                "model",
                HistGradientBoostingRegressor(
                    random_state=seed,
                    early_stopping=False,
                    max_iter=400,
                ),
            ),
        ]
    )
    # shallow trees + leaf restriction + L2 for ~100 parents.
    grid = {
        "model__max_depth": [2, 3],
        "model__max_leaf_nodes": [7, 15],
        "model__learning_rate": [0.03, 0.08],
        "model__min_samples_leaf": [10, 20],
        "model__l2_regularization": [0.0, 1.0],
    }
    return pipe, grid


def _kernel_ridge_rbf(seed: int):
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", KernelRidge(kernel="rbf")),
        ]
    )
    grid = {
        "model__alpha": [1e-2, 1e-1, 1.0],
        "model__gamma": [1e-3, 1e-2, 1e-1],
    }
    return pipe, grid


def _gaussian_process(seed: int):
    # Alternative kernel vs v3: Constant * RBF + WhiteKernel (explicit noise),
    # with bounded length-scale to regularise.
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e2))
        * RBF(length_scale=5.0, length_scale_bounds=(1e-1, 1e2))
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 1e2))
    )
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                GaussianProcessRegressor(
                    kernel=kernel,
                    normalize_y=True,
                    n_restarts_optimizer=2,
                    random_state=seed,
                    alpha=1e-6,
                ),
            ),
        ]
    )
    grid = {}  # kernel hyper-parameters are tuned by marginal likelihood
    return pipe, grid


def _extra_trees(seed: int):
    pipe = Pipeline(
        [
            ("scale", StandardScaler(with_mean=False, with_std=False)),
            (
                "model",
                ExtraTreesRegressor(
                    random_state=seed,
                    n_estimators=400,
                    n_jobs=1,
                ),
            ),
        ]
    )
    grid = {
        "model__max_depth": [4, 8, None],
        "model__min_samples_leaf": [2, 4],
        "model__max_features": [0.3, 0.6],
    }
    return pipe, grid


def _null_mean(seed: int):
    pipe = Pipeline([("scale", StandardScaler(with_mean=False, with_std=False)),
                     ("model", DummyRegressor(strategy="mean"))])
    return pipe, {}


MODEL_FAMILIES = {
    "huber": _huber,
    "bayesian_ridge": _bayesian_ridge,
    "huber_spline": _huber_spline,
    "bayesian_ridge_spline": _bayesian_ridge_spline,
    "ridge": _ridge,
    "hist_gbm": _hist_gbm,
    "kernel_ridge_rbf": _kernel_ridge_rbf,
    "gaussian_process": _gaussian_process,
    "extra_trees": _extra_trees,
    "null_mean": _null_mean,
}

# Recommended feature family per model (trainer default; overridable on CLI).
DEFAULT_FEATURE_FAMILY = {
    "huber": "physical",
    "bayesian_ridge": "physical",
    "huber_spline": "physical",
    "bayesian_ridge_spline": "physical",
    "ridge": "raw",
    "hist_gbm": "phys_raw",
    "kernel_ridge_rbf": "raw",
    "gaussian_process": "raw",
    "extra_trees": "phys_raw",
    "null_mean": "physical",
}

LINEAR_MODELS = {"huber", "bayesian_ridge", "ridge"}


def build_model(name: str, seed: int):
    if name not in MODEL_FAMILIES:
        raise KeyError(f"Unknown model {name!r}; choices: {sorted(MODEL_FAMILIES)}")
    return MODEL_FAMILIES[name](seed)


# --------------------------------------------------------------------------- #
# Modified Lucey/Kumar spectral-angle estimator.                               #
# --------------------------------------------------------------------------- #
from .features import BAND_WAVELENGTHS_NM, nearest_band  # noqa: E402


class SpectralAngleRegressor(BaseEstimator, RegressorMixin):
    """Lucey/Kumar ferrous spectral-angle model fitted on training data only.

    Given raw 74-band reflectance, computes R750 and R950 as band means, forms a
    spectral angle about an origin ``(x0, y0)`` in the (R750, R950/R750) plane,
    and maps a polynomial of that angle to the target.

        theta_i = arctan2( y0 - R950/R750_i ,  R750_i - x0 )
        y_pred  = sum_{k=0..degree} c_k * theta_i**k

    When ``fit_origin`` is False the origin is held at the published default and
    only the polynomial coefficients are least-squares fitted (modified slope /
    intercept).  When True, the origin is additionally optimised by minimising
    training MSE (Nelder-Mead over the 2 origin parameters, with the polynomial
    coefficients profiled out by least squares at each candidate origin).

    This is a *re-derived* model: nothing uses the dataset's precomputed
    Lucey/Kumar columns, so it can be trained inside a CV fold without leakage.
    """

    def __init__(self, degree: int = 1, fit_origin: bool = False,
                 x0: float = 0.08, y0: float = 1.19):
        self.degree = degree
        self.fit_origin = fit_origin
        self.x0 = x0
        self.y0 = y0

    @staticmethod
    def _bands(X):
        X = np.asarray(X, dtype=np.float64)
        lo7, hi7 = nearest_band(730.0), nearest_band(770.0)
        lo9, hi9 = nearest_band(930.0), nearest_band(970.0)
        r750 = X[:, lo7 : hi7 + 1].mean(axis=1)
        r950 = X[:, lo9 : hi9 + 1].mean(axis=1)
        return r750, r950

    def _theta(self, r750, r950, x0, y0):
        ratio = r950 / np.clip(r750, 1e-6, None)
        return np.arctan2(y0 - ratio, r750 - x0)

    def _design(self, theta):
        return np.vander(theta, N=self.degree + 1, increasing=True)

    def fit(self, X, y):
        y = np.asarray(y, dtype=np.float64)
        r750, r950 = self._bands(X)

        def profile(origin):
            x0, y0 = origin
            theta = self._theta(r750, r950, x0, y0)
            A = self._design(theta)
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            resid = y - A @ coef
            return float(resid @ resid), coef

        if self.fit_origin:
            from scipy.optimize import minimize

            best = None
            starts = [(self.x0, self.y0), (0.0, 1.0), (0.05, 1.3)]
            for s in starts:
                res = minimize(lambda o: profile(o)[0], s, method="Nelder-Mead",
                               options={"maxiter": 400, "xatol": 1e-4, "fatol": 1e-4})
                if best is None or res.fun < best.fun:
                    best = res
            self.x0_, self.y0_ = float(best.x[0]), float(best.x[1])
        else:
            self.x0_, self.y0_ = float(self.x0), float(self.y0)

        _, coef = profile((self.x0_, self.y0_))
        self.coef_ = coef
        return self

    def predict(self, X):
        r750, r950 = self._bands(X)
        theta = self._theta(r750, r950, self.x0_, self.y0_)
        return self._design(theta) @ self.coef_
