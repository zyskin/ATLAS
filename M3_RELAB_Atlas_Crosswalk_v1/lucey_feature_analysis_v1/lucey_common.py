#!/usr/bin/env python3
"""Shared constants and input helpers for Lucey-plane analyses."""

from __future__ import annotations

from pathlib import Path

import numpy as np


# Coordinates are (R750, R950/R750).
LUCEY_ORIGIN = (0.08, 1.19)
KUMAR_ORIGIN = (0.08, 1.18)


def read_tall_tsv(path: Path) -> np.ndarray:
    """Read one Stage-1 tall TSV while preserving its line/sample shape."""
    if not path.is_file():
        raise FileNotFoundError(path)
    array = np.loadtxt(path, delimiter="\t", dtype=np.float64, ndmin=2)
    if array.ndim != 2 or array.size == 0:
        raise ValueError(f"Expected a nonempty two-dimensional table: {path}")
    return array


def stage1_band_path(stage1_dir: Path, prefix: str, band: int) -> Path:
    return stage1_dir / f"{prefix}_band_{band}.tsv"


def load_stage1_lucey_coordinates(
    stage1_dir: Path,
    prefix: str,
    definition: str,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Return R750 and R950 for direct or Kumar three-band definitions."""
    if definition == "direct":
        low_bands = (9,)
        high_bands = (19,)
    elif definition == "kumar_mean":
        low_bands = (8, 9, 10)
        high_bands = (18, 19, 20)
    else:
        raise ValueError(f"Unknown band definition: {definition}")

    loaded: dict[int, np.ndarray] = {}
    shape: tuple[int, int] | None = None
    for band in (*low_bands, *high_bands):
        array = read_tall_tsv(stage1_band_path(stage1_dir, prefix, band))
        if shape is None:
            shape = array.shape
        elif array.shape != shape:
            raise ValueError(
                f"Band {band} has shape {array.shape}, but earlier bands have {shape}"
            )
        loaded[band] = array

    r750 = np.mean(np.stack([loaded[b] for b in low_bands]), axis=0)
    r950 = np.mean(np.stack([loaded[b] for b in high_bands]), axis=0)
    return r750, r950, (*low_bands, *high_bands)


def valid_lucey_values(
    r750: np.ndarray,
    r950: np.ndarray,
    reflectance_maximum: float,
    ratio_maximum: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select finite, positive, physically plottable reflectances."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = r950 / r750
    valid = (
        np.isfinite(r750)
        & np.isfinite(r950)
        & np.isfinite(ratio)
        & (r750 > 0.0)
        & (r950 >= 0.0)
        & (r750 <= reflectance_maximum)
        & (r950 <= reflectance_maximum)
        & (ratio >= 0.0)
        & (ratio <= ratio_maximum)
    )
    return r750[valid], ratio[valid], valid

