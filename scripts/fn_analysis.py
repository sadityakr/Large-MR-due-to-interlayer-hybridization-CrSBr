"""Fowler-Nordheim (FN) field-emission toolkit for the CrSBr vertical-transport data.

Lifts the FN analysis that was previously inlined in
`FN_five_points_20K.ipynb` into a reusable module so every per-temperature
"Spectroscopy and FN analysis" notebook and the T-dependence summary
notebook share one implementation (project rule: notebook orchestrates,
logic lives in scripts). The physics: in FN coordinates ln(|I|/V^2) vs 1/V,
direct tunnelling grows logarithmically while field emission falls
linearly; the crossover minimum is the transition voltage V_T, and in the
transition-voltage-spectroscopy picture eV_T estimates the barrier height
(Beebe 2006). The AFM endpoint gives a clean V_T (the coherent interlayer
channel is shut), which anchors the per-notebook calibration
c = V_peak^AFM / V_T^AFM used to convert conductance-peak positions into
barrier heights Phi = e*V_peak / c.

Input:
    IV_gaussian dataframes (one per T per axis) with columns 'H',
    'voltage_smooth', and 'current_smooth' (optionally
    'current_smooth_asym'). The SMU 1 nA noise floor is applied throughout.

Process:
    Average antisymmetric I(V) over a field window, transform to FN
    coordinates, locate the linear FN branch by an end-anchored R^2-grown
    fit, and find the V_T minimum. Field-binned variants sweep |H_z| to
    trace the canting trajectory.

Output:
    Pure functions returning numpy/pandas/dict; no file or plot side
    effects (the notebook owns plotting and saving).
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=2.0",
#   "pandas>=2.0",
# ]
# ///
from __future__ import annotations

import numpy as np
import pandas as pd

I_NOISE = 1.0e-9      # A; SMU noise floor (drop |I| below this everywhere)
H_AFM_MAX = 0.10      # T; |H_z| window that defines the AFM endpoint

# Per-T FM-saturation cutoffs. H_sat^c falls from ~2.20 T (20 K) toward
# ~1.45 T near T_N (Lopez-Paz 2022). Each entry sits ~0.10 T below the
# local H_sat so the FM window is genuinely saturated.
H_FM_MIN_BY_T = {
    20: 2.10, 30: 2.05, 40: 1.95, 50: 1.85,
    60: 1.75, 70: 1.65, 80: 1.55, 90: 1.45, 100: 1.35,
}


def fm_cutoff(T):
    """Lower |H_z| bound of the saturated-FM window at temperature T (K)."""
    if T in H_FM_MIN_BY_T:
        return H_FM_MIN_BY_T[T]
    return max(2.20 + (1.45 - 2.20) * (T - 20) / 80.0 - 0.10, 1.20)


def average_iv(df, mask, n_grid=401, V_min=0.005, V_max=0.95,
               current_col="current_smooth_asym"):
    """Average antisymmetric I(V) over rows matching `mask` on a uniform V grid.

    Falls back to 'current_smooth' if 'current_smooth_asym' is absent.
    Returns (V_grid, I_mean, n_curves); I_mean is None if no curve qualifies.
    """
    if current_col not in df.columns:
        current_col = "current_smooth"
    V_grid = np.linspace(V_min, V_max, n_grid)
    curves = []
    for _, r in df.loc[mask].iterrows():
        V = np.asarray(r["voltage_smooth"])
        I = np.asarray(r[current_col])
        order = np.argsort(V)
        V, I = V[order], I[order]
        m_pos = V > 0
        if m_pos.sum() < 10:
            continue
        curves.append(np.interp(V_grid, V[m_pos], I[m_pos],
                                left=np.nan, right=np.nan))
    if not curves:
        return V_grid, None, 0
    A = np.vstack(curves)
    return V_grid, np.nanmean(A, axis=0), A.shape[0]


def fn_transform(V, I, I_floor=I_NOISE):
    """Map (V, I) to FN coordinates: x = 1/V, y = ln(|I|/V^2), sorted by x.

    Returns (x, y, Vp) where Vp is the matching bias array. Points with
    V <= 0 or |I| below the noise floor are dropped.
    """
    mask = (V > 0) & np.isfinite(I) & (np.abs(I) >= I_floor)
    Vp = V[mask]
    Ip = I[mask]
    x = 1.0 / Vp
    y = np.log(np.abs(Ip) / Vp**2)
    order = np.argsort(x)
    return x[order], y[order], Vp[order]


def find_local_min(y, smooth=5):
    """Index of the first local minimum of y (the FN transition), or None.

    y must be ordered by increasing 1/V (decreasing V). A boxcar smooth of
    width `smooth` suppresses point noise before the sign-change scan.
    """
    if y.size < smooth + 2:
        return None
    pad = smooth // 2
    yk = np.array([np.mean(y[max(0, i - pad):min(len(y), i + pad + 1)])
                   for i in range(len(y))])
    dy = np.diff(yk)
    for i in range(1, len(dy)):
        if dy[i - 1] <= 0 < dy[i]:
            return i
    return None


def fit_fn(V, I, R2_thr=0.998, min_pts=10):
    """End-anchored linear-region fit on the FN plot.

    Grows a contiguous window from the high-V end (smallest 1/V) outward,
    keeping it while R^2 stays above `R2_thr`. The last accepted window
    defines the FN slope. Returns None if no window meets the threshold
    (the cleanest no-FN-regime signal, e.g. the FM endpoint where the
    coherent interlayer channel fills in the knee).

    Result dict keys: B_FN (= -slope), slope, intercept, R2, V_T (NaN if no
    minimum resolves), i_window, x, y, Vp.
    """
    x, y, Vp = fn_transform(V, I)
    if len(x) < min_pts:
        return None
    best = None
    for i1 in range(min_pts, len(x) + 1):
        xs, ys = x[:i1], y[:i1]
        slope, intercept = np.polyfit(xs, ys, 1)
        yhat = slope * xs + intercept
        ss_res = float(np.sum((ys - yhat)**2))
        ss_tot = float(np.sum((ys - ys.mean())**2))
        R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        if R2 >= R2_thr:
            best = (i1, slope, intercept, R2)
        else:
            break
    if best is None:
        return None
    i1, slope, intercept, R2 = best
    i_loc = find_local_min(y)
    V_T = float(Vp[i_loc]) if i_loc is not None else float("nan")
    return dict(B_FN=-float(slope), slope=float(slope), intercept=float(intercept),
                R2=float(R2), V_T=V_T, i_window=(0, i1),
                x=x, y=y, Vp=Vp)


def endpoint_curves(df, T, h_afm_max=H_AFM_MAX, fm_min=None, **avg_kwargs):
    """Average AFM- and FM-endpoint I(V) for one (T, axis) dataframe.

    fm_min defaults to fm_cutoff(T). Returns (V, I_AFM, I_FM, n_AFM, n_FM).
    """
    if fm_min is None:
        fm_min = fm_cutoff(T)
    V, I_AFM, n_A = average_iv(df, df["H"].abs() < h_afm_max, **avg_kwargs)
    _, I_FM, n_F = average_iv(df, df["H"].abs() > fm_min, **avg_kwargs)
    return V, I_AFM, I_FM, n_A, n_F


def field_binned_fn(df, h_bin_centers, bin_hw=0.10, **avg_kwargs):
    """FN fit for each |H_z| bin, tracing the canting trajectory.

    Returns a list of dicts, one per bin center, each with keys:
    H_center, n_curves, B_FN, V_T, fit (the full fit_fn result or None),
    x, y (FN-coordinate arrays for plotting, or None).
    """
    out = []
    for Hc in h_bin_centers:
        if Hc == 0.0:
            mask = np.abs(df["H"]) < bin_hw
        else:
            mask = np.abs(np.abs(df["H"]) - Hc) < bin_hw
        V_grid, I_mean, n_curves = average_iv(df, mask, **avg_kwargs)
        if I_mean is None:
            out.append(dict(H_center=Hc, n_curves=0, B_FN=np.nan, V_T=np.nan,
                            fit=None, x=None, y=None))
            continue
        x, y, _ = fn_transform(V_grid, I_mean)
        fr = fit_fn(V_grid, I_mean)
        out.append(dict(
            H_center=Hc, n_curves=n_curves,
            B_FN=fr["B_FN"] if fr is not None else np.nan,
            V_T=fr["V_T"] if fr is not None else np.nan,
            fit=fr, x=x, y=y,
        ))
    return out


def calibration_factor(v_peak_afm, v_t_afm):
    """Dimensionless TVS calibration c = V_peak^AFM / V_T^AFM.

    Both observables are measured in the AFM endpoint of one notebook
    (one T, one axis), so c carries no model input. Phi = e*V_peak / c then
    transfers this calibration to every field, with Phi_AFM = e*V_T^AFM by
    construction. Returns NaN if V_T^AFM is not finite or non-positive.
    """
    if not np.isfinite(v_t_afm) or v_t_afm <= 0:
        return float("nan")
    return float(v_peak_afm) / float(v_t_afm)


def phi_from_vpeak(v_peak, c):
    """Barrier height Phi (eV) from conductance-peak position V_peak (V).

    Phi = V_peak / c. Vectorised over numpy arrays / pandas Series.
    """
    return np.asarray(v_peak, dtype=float) / float(c)


__all__ = [
    "I_NOISE", "H_AFM_MAX", "H_FM_MIN_BY_T", "fm_cutoff",
    "average_iv", "fn_transform", "find_local_min", "fit_fn",
    "endpoint_curves", "field_binned_fn",
    "calibration_factor", "phi_from_vpeak",
]
