# ---
# description: |
#   Helpers for estimating the c-axis saturation field H_sat and the
#   b-axis spin-flip field H_sf from IV_Hscan dataframes. H_sat is fit
#   from I(|H_z|) at a chosen constant bias V_bias using a hinge model
#   (linear + plateau). H_sf is fit from I(|H_y|) using a tanh sigmoid.
#   Both helpers also offer a cross-check from the Phi(H) plateau curve
#   so that the user can flag disagreement between the I(H)- and
#   Phi(H)-based estimates.
# entry_point: from scripts.saturation_field import estimate_H_sat_c, estimate_H_sf_b
# dependencies:
#   - numpy
#   - scipy.optimize
# input: |
#   estimate_H_sat_c expects a c-axis IV_gaussian dataframe with columns
#   'H', 'voltage_smooth', 'current_smooth'. estimate_H_sf_b expects the
#   same for a b-axis dataframe.
#   For cross-check (cross_check_with_Phi), the caller passes the per-curve
#   binned Phi(|H|) table from the notebook.
# process: |
#   estimate_H_sat_c:
#     1. Interpolate I(V_bias) for each curve in the dataframe.
#     2. Average symmetric branches: I_avg(|H|) = 0.5 * (I(+H) + I(-H)).
#     3. Fit a hinge function I_avg(h) = I0 + s * min(h, H_sat) using
#        scipy.optimize.curve_fit. Return H_sat with cov-based error.
#   estimate_H_sf_b:
#     1. Same I_avg(|H|) construction.
#     2. Fit tanh sigmoid I_avg(h) = a + b * tanh((h - H_sf) / w) / 2.
#        Return H_sf with cov-based error.
#   cross_check_with_Phi: compare a precomputed plateau breakpoint in
#   Phi(|H|) to the I-based estimate.
# output: |
#   Each estimator returns a dict with keys H_sat (or H_sf), H_err, popt,
#   pcov, fit_h, fit_I (for plotting), V_bias, and ok (bool).
# last_updated: 2026-05-21
# ---
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _interp_I_at_V(V_arr, I_arr, V_target):
    order = np.argsort(V_arr)
    V_s, I_s = np.asarray(V_arr)[order], np.asarray(I_arr)[order]
    return float(np.interp(V_target, V_s, I_s))


def _build_I_vs_absH(df, V_bias, V_col='voltage_smooth', I_col='current_smooth'):
    """Average symmetric H-branches at the requested V_bias."""
    H = df['H'].values
    I = np.array([_interp_I_at_V(r[V_col], r[I_col], V_bias) for _, r in df.iterrows()])
    abs_H = np.abs(H)
    # Bin to 10 mT, mean within bin.
    key = np.round(abs_H, 2)
    order = np.argsort(key)
    key, I_o, abs_o = key[order], I[order], abs_H[order]
    uniq, inv = np.unique(key, return_inverse=True)
    I_avg = np.zeros_like(uniq)
    h_avg = np.zeros_like(uniq)
    counts = np.zeros_like(uniq)
    for j in range(len(uniq)):
        sel = inv == j
        I_avg[j] = I_o[sel].mean()
        h_avg[j] = abs_o[sel].mean()
        counts[j] = sel.sum()
    return h_avg, I_avg, counts


# ---------------------------------------------------------------------------
# c-axis: hinge fit for H_sat
# ---------------------------------------------------------------------------
def _hinge(h, I0, slope, H_sat):
    return I0 + slope * np.minimum(h, H_sat)


def estimate_H_sat_c(df, V_bias=0.6, H_sat_init=2.0,
                     V_col='voltage_smooth', I_col='current_smooth',
                     H_max=None):
    """
    Estimate the c-axis saturation field by fitting I(|H_z|) at V = V_bias to
    a hinge model: I rises linearly with |H| until |H| = H_sat, then flattens.

    Parameters
    ----------
    df : pandas.DataFrame
        c-axis IV_gaussian dataframe.
    V_bias : float
        Bias voltage at which to evaluate I (V). Pick a value where the
        AFM-to-FM contrast in I is large, typically 0.5-0.7 V for CrSBr.
    H_sat_init : float
        Initial guess for H_sat (T). SQUID reference value at 20 K is 2.0 T.
    H_max : float or None
        If set, fit only |H_z| <= H_max. Useful to suppress noisy points
        well above saturation.

    Returns
    -------
    dict with keys:
        H_sat        : best-fit saturation field (T)
        H_sat_err    : 1-sigma error from covariance (T)
        I0, slope    : nuisance parameters
        popt, pcov   : raw curve_fit output
        h, I_avg     : data used for the fit (after symmetrization & binning)
        fit_h, fit_I : dense model curve for plotting
        V_bias       : echoed input
        ok           : True if the fit converged with H_sat strictly inside
                       the bounded interval (0, max(|H|))
    """
    h, I_avg, _ = _build_I_vs_absH(df, V_bias, V_col=V_col, I_col=I_col)
    if H_max is not None:
        m = h <= H_max
        h, I_avg = h[m], I_avg[m]

    if h.size < 4:
        return dict(H_sat=np.nan, H_sat_err=np.nan, popt=None, pcov=None,
                    h=h, I_avg=I_avg, fit_h=None, fit_I=None, V_bias=V_bias,
                    ok=False)

    H_lo, H_hi = float(h.min() + 1e-3), float(h.max())
    I0_init = float(I_avg.min())
    slope_init = float((I_avg.max() - I_avg.min()) / max(H_hi - H_lo, 1e-3))
    p0 = [I0_init, slope_init, H_sat_init]
    lower = [-np.inf, -np.inf, H_lo]
    upper = [np.inf,  np.inf,  H_hi]
    try:
        popt, pcov = curve_fit(_hinge, h, I_avg, p0=p0, bounds=(lower, upper),
                               maxfev=20000)
    except Exception:
        return dict(H_sat=np.nan, H_sat_err=np.nan, popt=None, pcov=None,
                    h=h, I_avg=I_avg, fit_h=None, fit_I=None, V_bias=V_bias,
                    ok=False)

    I0, slope, H_sat = popt
    perr = np.sqrt(np.diag(pcov))
    fit_h = np.linspace(h.min(), h.max(), 200)
    fit_I = _hinge(fit_h, *popt)
    eps = 5e-3
    at_bound = (H_sat <= H_lo + eps) or (H_sat >= H_hi - eps)
    return dict(H_sat=float(H_sat), H_sat_err=float(perr[2]),
                I0=float(I0), slope=float(slope),
                popt=popt, pcov=pcov,
                h=h, I_avg=I_avg, fit_h=fit_h, fit_I=fit_I,
                V_bias=V_bias, ok=not at_bound)


# ---------------------------------------------------------------------------
# b-axis: tanh sigmoid for H_sf
# ---------------------------------------------------------------------------
def _tanh_step(h, a, b, H_sf, w):
    return a + 0.5 * b * np.tanh((h - H_sf) / w)


def estimate_H_sf_b(df, V_bias=0.6, H_sf_init=0.3, w_init=0.05,
                    V_col='voltage_smooth', I_col='current_smooth',
                    H_max=None):
    """
    Estimate the b-axis spin-flip field by fitting I(|H_y|) at V = V_bias
    to a tanh sigmoid centred at H_sf with width w.

    Returns the same dict shape as estimate_H_sat_c, with the field key
    renamed to H_sf and H_sf_err.
    """
    h, I_avg, _ = _build_I_vs_absH(df, V_bias, V_col=V_col, I_col=I_col)
    if H_max is not None:
        m = h <= H_max
        h, I_avg = h[m], I_avg[m]

    if h.size < 5:
        return dict(H_sf=np.nan, H_sf_err=np.nan, popt=None, pcov=None,
                    h=h, I_avg=I_avg, fit_h=None, fit_I=None, V_bias=V_bias,
                    ok=False)

    H_lo, H_hi = float(h.min() + 1e-3), float(h.max())
    a_init = float(0.5 * (I_avg.min() + I_avg.max()))
    b_init = float(I_avg.max() - I_avg.min())
    p0 = [a_init, b_init, H_sf_init, w_init]
    lower = [-np.inf, -np.inf, H_lo,  1e-3]
    upper = [np.inf,   np.inf, H_hi,  0.3]
    try:
        popt, pcov = curve_fit(_tanh_step, h, I_avg, p0=p0, bounds=(lower, upper),
                               maxfev=20000)
    except Exception:
        return dict(H_sf=np.nan, H_sf_err=np.nan, popt=None, pcov=None,
                    h=h, I_avg=I_avg, fit_h=None, fit_I=None, V_bias=V_bias,
                    ok=False)

    a, b, H_sf, w = popt
    perr = np.sqrt(np.diag(pcov))
    fit_h = np.linspace(h.min(), h.max(), 200)
    fit_I = _tanh_step(fit_h, *popt)
    eps = 5e-3
    at_bound = (H_sf <= H_lo + eps) or (H_sf >= H_hi - eps)
    return dict(H_sf=float(H_sf), H_sf_err=float(perr[2]),
                a=float(a), b=float(b), w=float(w),
                popt=popt, pcov=pcov,
                h=h, I_avg=I_avg, fit_h=fit_h, fit_I=fit_I,
                V_bias=V_bias, ok=not at_bound)


# ---------------------------------------------------------------------------
# Cross-check from Phi(|H|) plateau
# ---------------------------------------------------------------------------
def cross_check_phi_plateau(abs_H, Phi_meV, Phi_err_meV,
                            tol_meV=10.0, min_consecutive=3):
    """
    Diagnostic: walk down |H| from the maximum field and find the smallest
    |H| at which the binned Phi(|H|) is within tol_meV of the FM-saturated
    Phi_FM (taken as the median Phi over the top quartile in |H|). Returns
    that breakpoint as a Phi-based estimate of H_sat to compare against
    the I(H)-based estimate.
    """
    abs_H = np.asarray(abs_H, float)
    Phi = np.asarray(Phi_meV, float)
    order = np.argsort(abs_H)
    abs_H, Phi = abs_H[order], Phi[order]
    n = len(abs_H)
    if n < 5:
        return dict(H_break=np.nan, Phi_FM_proxy_meV=np.nan, ok=False)

    top = max(1, n // 4)
    Phi_FM_proxy = float(np.median(Phi[-top:]))
    consec = 0
    H_break = np.nan
    for i in range(n - 1, -1, -1):
        if abs(Phi[i] - Phi_FM_proxy) <= tol_meV:
            consec += 1
            if consec >= min_consecutive:
                H_break = float(abs_H[i])
        else:
            consec = 0
    return dict(H_break=H_break, Phi_FM_proxy_meV=Phi_FM_proxy,
                ok=np.isfinite(H_break))


__all__ = ['estimate_H_sat_c', 'estimate_H_sf_b', 'cross_check_phi_plateau']
