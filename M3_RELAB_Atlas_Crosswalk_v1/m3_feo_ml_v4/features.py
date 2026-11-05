"""Physically-motivated spectral feature families for M3-convolved RELAB spectra.

This module defines, for the 74 M3 global bandpasses present in the v2 ground
truth table (columns ``m3_b01`` ... ``m3_b74``), several feature families built
only from reflectance and the *known* band-centre wavelengths.  No target
information enters any feature.  Every transform is a pure function of a
reflectance matrix ``X`` (n_samples x 74), so it is safe to apply inside a
cross-validation fold.

The band-centre wavelengths are taken verbatim from the M3 forward model used to
build the dataset (``relab_to_m3_bandpasses.py`` -> ``relab_m3_spectra.csv``;
``m3_response_centroid_nm`` for M3 global bands 1..74).  Band 9 == 750.44 nm and
band 19 == 950.06 nm were cross-checked against the dataset columns
``r750_direct_b09`` and ``r950_direct_b19``.

Feature families
----------------
``raw``       : the 74 reflectance bands, unchanged (baseline input).
``physical``  : a compact set of scientifically chosen spectral descriptors
                (visible-NIR slopes, band means/ratios/log-ratios, a 1 micron
                band depth with linear continuum removal, selected band
                contrasts, and a global reflectance level).
``derivative``: first and second spectral derivatives (finite differences w.r.t.
                wavelength) summarised on a coarse grid, capturing local slope
                and curvature without the full 74-d noise.
``shape``     : continuum-removed / normalised shape vector (reflectance divided
                by a straight-line continuum between the 700 and 1600 nm anchors)
                plus the raw bands, emphasising absorption shape over albedo.
``phys_raw``  : ``physical`` descriptors concatenated with the raw bands.

All families return a 2-D float array plus a list of human-readable feature
names (same length as the number of columns), so downstream code can report
standardized coefficients and block-permutation importance by name.
"""
from __future__ import annotations

import numpy as np

# M3 global band centres (nm) for bands 1..74, from relab_m3_spectra.csv.
BAND_WAVELENGTHS_NM: np.ndarray = np.array(
    [
        460.9900, 500.9200, 540.8400, 580.7650, 620.6900, 660.6100, 700.5375,
        730.4800, 750.4400, 770.4000, 790.3650, 810.3300, 830.2900, 850.2500,
        870.2100, 890.1750, 910.1400, 930.1000, 950.0600, 970.0200, 989.9800,
        1009.9500, 1029.9100, 1049.8700, 1069.8300, 1089.7900, 1109.7600,
        1129.7200, 1149.6800, 1169.6400, 1189.6000, 1209.5700, 1229.5300,
        1249.4900, 1269.4500, 1289.4100, 1309.3750, 1329.3400, 1349.3000,
        1369.2600, 1389.2200, 1409.1850, 1429.1500, 1449.1100, 1469.0700,
        1489.0300, 1508.9950, 1528.9600, 1548.9200, 1578.8600, 1618.7875,
        1658.7100, 1698.6300, 1738.5600, 1778.4800, 1818.4050, 1858.3300,
        1898.2500, 1938.1800, 1978.1000, 2018.0225, 2057.9500, 2097.8700,
        2137.7975, 2177.7200, 2217.6425, 2257.5700, 2297.4900, 2337.4150,
        2377.3400, 2417.2600, 2457.1900, 2497.1100, 2537.0350,
    ],
    dtype=np.float64,
)

N_BANDS = BAND_WAVELENGTHS_NM.size
assert N_BANDS == 74

BAND_COLUMNS = [f"m3_b{i:02d}" for i in range(1, N_BANDS + 1)]


def nearest_band(wavelength_nm: float) -> int:
    """Index (0-based) of the band whose centre is nearest ``wavelength_nm``."""
    return int(np.argmin(np.abs(BAND_WAVELENGTHS_NM - wavelength_nm)))


# Pre-computed anchor band indices used by several descriptors.
_B460 = nearest_band(460.0)
_B560 = nearest_band(560.0)
_B700 = nearest_band(700.0)
_B750 = nearest_band(750.0)   # == band 9
_B900 = nearest_band(900.0)
_B950 = nearest_band(950.0)   # == band 19
_B1000 = nearest_band(1000.0)
_B1050 = nearest_band(1050.0)
_B1250 = nearest_band(1250.0)
_B1580 = nearest_band(1580.0)
_B1600 = nearest_band(1600.0)
_B1900 = nearest_band(1900.0)
_B2000 = nearest_band(2000.0)
_B2500 = nearest_band(2500.0)

_EPS = 1e-6


def _safe_log_ratio(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    """log(numer/denom) guarded for non-positive reflectance."""
    num = np.clip(numer, _EPS, None)
    den = np.clip(denom, _EPS, None)
    return np.log(num / den)


def _band_mean(X: np.ndarray, lo_nm: float, hi_nm: float) -> np.ndarray:
    lo = nearest_band(lo_nm)
    hi = nearest_band(hi_nm)
    lo, hi = min(lo, hi), max(lo, hi)
    return X[:, lo : hi + 1].mean(axis=1)


def physical_features(X: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Compact, physically interpretable spectral descriptors.

    Designed for mare/highland lunar soils: continuum slope (space weathering /
    composition), ferrous 1 micron band depth and position proxy, UV/VIS ratio,
    and broad NIR behaviour.  All are intensive (ratios/slopes) except the
    overall reflectance level, which is kept once.
    """
    cols: list[np.ndarray] = []
    names: list[str] = []

    wl = BAND_WAVELENGTHS_NM

    def add(name: str, values: np.ndarray) -> None:
        cols.append(np.asarray(values, dtype=np.float64))
        names.append(name)

    # Overall reflectance level (albedo proxy) at ~1500 nm continuum.
    add("refl_level_1500", _band_mean(X, 1450.0, 1550.0))

    # Visible-to-NIR continuum slope: log-reflectance regressed on wavelength
    # over 700-1600 nm (outside strong absorptions), per sample.
    vis_nir_lo = nearest_band(700.0)
    vis_nir_hi = nearest_band(1600.0)
    seg_wl = wl[vis_nir_lo : vis_nir_hi + 1]
    seg = np.clip(X[:, vis_nir_lo : vis_nir_hi + 1], _EPS, None)
    w_centered = seg_wl - seg_wl.mean()
    denom = float(np.sum(w_centered ** 2))
    log_seg = np.log(seg)
    slope_vis_nir = (log_seg - log_seg.mean(axis=1, keepdims=True)) @ w_centered / denom
    add("logslope_700_1600_per_nm", slope_vis_nir)

    # Short-visible slope (UV-VIS continuum, maturity/composition sensitive).
    short_lo = nearest_band(460.0)
    short_hi = nearest_band(750.0)
    seg_wl2 = wl[short_lo : short_hi + 1]
    seg2 = np.clip(X[:, short_lo : short_hi + 1], _EPS, None)
    w2 = seg_wl2 - seg_wl2.mean()
    log_seg2 = np.log(seg2)
    slope_vis = (log_seg2 - log_seg2.mean(axis=1, keepdims=True)) @ w2 / float(np.sum(w2 ** 2))
    add("logslope_460_750_per_nm", slope_vis)

    # Classic band means and ratios (echoing Lucey/Kumar anchors, but as inputs
    # not target-derived predictions).
    r750 = _band_mean(X, 730.0, 770.0)
    r950 = _band_mean(X, 930.0, 970.0)
    r1500 = _band_mean(X, 1450.0, 1550.0)
    r2000 = _band_mean(X, 1960.0, 2040.0)
    add("ratio_r950_r750", r950 / np.clip(r750, _EPS, None))
    add("logratio_r950_r750", _safe_log_ratio(r950, r750))
    add("ratio_r1500_r750", r1500 / np.clip(r750, _EPS, None))
    add("logratio_r2000_r1500", _safe_log_ratio(r2000, r1500))

    # 1 micron ferrous band depth with a linear continuum removal between the
    # 700 nm and 1600 nm shoulders evaluated at the band minimum near ~950-1000.
    lo = nearest_band(700.0)
    hi = nearest_band(1600.0)
    wl_lo, wl_hi = wl[lo], wl[hi]
    r_lo, r_hi = X[:, lo], X[:, hi]
    band_idx = np.arange(nearest_band(850.0), nearest_band(1100.0) + 1)
    # linear continuum value at each wavelength in the 1 um region
    frac = (wl[band_idx][None, :] - wl_lo) / (wl_hi - wl_lo)
    continuum = r_lo[:, None] * (1.0 - frac) + r_hi[:, None] * frac
    depth = 1.0 - X[:, band_idx] / np.clip(continuum, _EPS, None)
    add("band1um_max_depth", depth.max(axis=1))
    add("band1um_mean_depth", depth.mean(axis=1))
    # crude band-centre position proxy (wavelength of deepest point)
    pos = wl[band_idx][np.argmax(depth, axis=1)]
    add("band1um_center_nm", pos)

    # 2 micron region behaviour (pyroxene second band shoulder, weak at M3 global
    # sampling but informative for composition).
    add("logratio_r2500_r2000", _safe_log_ratio(_band_mean(X, 2460.0, 2537.0), r2000))

    # Selected visible band contrasts.
    add("contrast_560_460", _safe_log_ratio(_band_mean(X, 540.0, 580.0), _band_mean(X, 460.0, 500.0)))

    M = np.column_stack(cols)
    return M, names


def derivative_features(X: np.ndarray, step: int = 4) -> tuple[np.ndarray, list[str]]:
    """First and second spectral derivatives summarised on a coarse grid.

    Derivatives are finite differences w.r.t. wavelength; summarising every
    ``step`` bands keeps dimensionality modest for ~100 parents.
    """
    wl = BAND_WAVELENGTHS_NM
    # first derivative between consecutive bands
    d1 = np.diff(X, axis=1) / np.diff(wl)[None, :]
    # second derivative
    d2 = np.diff(d1, axis=1) / np.diff(wl[:-1])[None, :]
    cols = []
    names = []
    for j in range(0, d1.shape[1], step):
        cols.append(d1[:, j])
        names.append(f"d1_b{j + 1:02d}")
    for j in range(0, d2.shape[1], step):
        cols.append(d2[:, j])
        names.append(f"d2_b{j + 1:02d}")
    M = np.column_stack(cols)
    return M, names


def shape_features(X: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Continuum-removed shape (albedo-insensitive) + raw bands.

    Each band reflectance is divided by a straight-line continuum between the
    700 nm and 1600 nm anchors, isolating absorption shape from overall
    brightness; the raw bands are appended so brightness is still available.
    """
    wl = BAND_WAVELENGTHS_NM
    lo = nearest_band(700.0)
    hi = nearest_band(1600.0)
    r_lo, r_hi = X[:, lo], X[:, hi]
    frac = (wl[None, :] - wl[lo]) / (wl[hi] - wl[lo])
    continuum = r_lo[:, None] * (1.0 - frac) + r_hi[:, None] * frac
    shape = X / np.clip(continuum, _EPS, None)
    names = [f"shape_b{i:02d}" for i in range(1, N_BANDS + 1)]
    names += [f"m3_b{i:02d}" for i in range(1, N_BANDS + 1)]
    M = np.column_stack([shape, X])
    return M, names


def raw_features(X: np.ndarray) -> tuple[np.ndarray, list[str]]:
    return X.astype(np.float64), list(BAND_COLUMNS)


def phys_raw_features(X: np.ndarray) -> tuple[np.ndarray, list[str]]:
    phys, pn = physical_features(X)
    return np.column_stack([phys, X]), pn + list(BAND_COLUMNS)


FEATURE_FAMILIES = {
    "raw": raw_features,
    "physical": physical_features,
    "derivative": derivative_features,
    "shape": shape_features,
    "phys_raw": phys_raw_features,
}


def build_features(X: np.ndarray, family: str) -> tuple[np.ndarray, list[str]]:
    if family not in FEATURE_FAMILIES:
        raise KeyError(f"Unknown feature family {family!r}; choices: {sorted(FEATURE_FAMILIES)}")
    return FEATURE_FAMILIES[family](np.asarray(X, dtype=np.float64))


def feature_block_indices(names: list[str]) -> dict[str, list[int]]:
    """Group feature columns into interpretable blocks for permutation importance.

    For raw / shape families, adjacent bands are grouped into fixed-width blocks
    of ~8 bands (so permutation perturbs a *region* of the spectrum, not a single
    correlated channel).  Named physical/derivative features are grouped by their
    semantic prefix.
    """
    blocks: dict[str, list[int]] = {}
    band_like = [i for i, n in enumerate(names) if n.startswith(("m3_b", "shape_b"))]
    # fixed adjacent blocks of 8 for band-like features, keyed by prefix+range
    width = 8
    for group_prefix in ("m3_b", "shape_b"):
        idxs = [i for i in band_like if names[i].startswith(group_prefix)]
        for start in range(0, len(idxs), width):
            chunk = idxs[start : start + width]
            if not chunk:
                continue
            lo_b = names[chunk[0]]
            hi_b = names[chunk[-1]]
            blocks[f"{group_prefix}block_{lo_b}_{hi_b}"] = chunk
    # everything else grouped by semantic prefix token
    for i, n in enumerate(names):
        if n.startswith(("m3_b", "shape_b")):
            continue
        key = n
        for pref in ("logslope", "logratio", "ratio", "band1um", "d1_", "d2_", "contrast", "refl_level"):
            if n.startswith(pref):
                key = pref.rstrip("_")
                break
        blocks.setdefault(key, []).append(i)
    return blocks
