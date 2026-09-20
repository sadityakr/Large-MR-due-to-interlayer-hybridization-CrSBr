# ---
# description: |
#   Two-step Gaussian + linear background peak fitter for the
#   Feenstra-normalised differential conductance (dI/dV)/(I/V), plus
#   inverse-variance weighted mean and field-binning utilities. The fit
#   is explicit about its two phases:
#     Step 1 (rough): savgol-smoothed argmax in the coarse window gives
#       a starting V_rough; a wide Gaussian + linear fit then refines it
#       to V_peak_rough.
#     Step 2 (tight): Gaussian + linear refit in a narrow window
#       V_peak_rough +/- tight_half_width where the data is approximately
#       Gaussian. The tight-fit sigma and V_peak are the reported values.
#   The sigma upper bound rejection used in the 20K-only v2 notebook is
#   removed so that broader peaks at higher T survive and their sigma can
#   be reported (the band-smudging story).
# entry_point: from scripts.barrier_canting_fit import (
#       normalised_didv, fit_band_edge_gauss, gauss_lin, extract_peaks,
#       weighted_mean, bin_weighted, canting_angle, sin2_half)
# dependencies:
#   - numpy, pandas, scipy
# input: |
#   normalised_didv: voltage_smooth and current_smooth from one IV curve.
#   fit_band_edge_gauss: V, norm arrays. extract_peaks: IV_gaussian dataframe.
# process: |
#   Compute the normalised dI/dV (with an outlier clip on |norm| > 20 to
#   drop the I-near-zero blow-ups seen at higher T). Two-step Gaussian +
#   linear fit as described above. Acceptance criterion: both fits
#   converged, V_peak inside the tight window interior, sigma > 0.015,
#   sigma_err finite. Upper-bound rejection on sigma is opt-in via
#   `reject_sigma_upper=True` (recovers the strict 20 K-only behaviour).
#   extract_peaks forwards **fit_kwargs to fit_band_edge_gauss so callers
#   can tighten sigma_bounds_tight per T.
# output: |
#   Pure functions returning numpy/pandas. Fit result dict has keys
#   V_peak, V_peak_err, sigma, sigma_err, ok, popt, window, V_rough.
# last_updated: 2026-05-22
# ---
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

# ---- Default fit parameters (validated across 20-100 K T-sweep) ----
COARSE_WINDOW       = (0.40, 0.98)
ROUGH_HALF_WIDTH    = 0.30   # wide window for the rough Gaussian
TIGHT_HALF_WIDTH    = 0.18   # narrow window for the tight Gaussian
SIGMA_LO            = 0.015
SIGMA_HI_ROUGH      = 0.60   # generous on rough fit
SIGMA_HI_TIGHT      = 0.60   # do NOT bound-reject on this any more
SMOOTH_WIN, SMOOTH_POLY = 21, 3
TRIM, V_MIN         = 2, 0.05
NORM_ABS_MAX        = 20.0   # drop points where (dI/dV)/(I/V) blows up
V_MAX_MARGIN        = 0.02   # also drop points within this V of the data tail

# Backwards-compat aliases used by the 20K v2 notebook.
HALF_WIDTH          = TIGHT_HALF_WIDTH
SIGMA_BOUNDS        = (SIGMA_LO, SIGMA_HI_TIGHT)


def normalised_didv(V, I, trim=TRIM, v_min=V_MIN,
                    norm_abs_max=NORM_ABS_MAX, v_max_margin=V_MAX_MARGIN):
    """Sort by V, compute dI/dV by gradient, return (V, norm) where
    norm = (dI/dV) / (I / V). Drops |V| < v_min, drops `trim` points from
    each end, drops points within v_max_margin of the V tail (to suppress
    the boundary noise from differentiation), and drops any |norm| beyond
    norm_abs_max (which would be a numerical blow-up from I crossing zero).
    """
    order = np.argsort(V)
    V, I = np.asarray(V)[order], np.asarray(I)[order]
    dIdV = np.gradient(I, V)
    V, I, dIdV = V[trim:-trim], I[trim:-trim], dIdV[trim:-trim]

    mask = np.abs(V) > v_min
    V, I, dIdV = V[mask], I[mask], dIdV[mask]
    if V.size == 0:
        return V, dIdV
    V_max = V.max()
    keep_v = V <= V_max - v_max_margin
    V, I, dIdV = V[keep_v], I[keep_v], dIdV[keep_v]

    norm = dIdV / (I / V)
    keep = np.abs(norm) < norm_abs_max
    return V[keep], norm[keep]


def gauss_lin(V, A, V0, sigma, m, c):
    return A * np.exp(-0.5 * ((V - V0) / sigma) ** 2) + m * V + c


def _fit_gauss_in_window(Vp, Np_, V_center, half_width, sigma_bounds):
    """Inner fit primitive. Returns (popt, perr, V_peak_grid, ok)."""
    n_edge = max(3, len(Vp) // 6)
    m_init = (Np_[-n_edge:].mean() - Np_[:n_edge].mean()) / (Vp[-n_edge:].mean() - Vp[:n_edge].mean())
    c_init = Np_[:n_edge].mean() - m_init * Vp[:n_edge].mean()
    A_init = max(float(Np_.max() - (m_init * V_center + c_init)), 1e-3)
    p0 = [A_init, V_center, 0.06, m_init, c_init]
    lower = [0,                     V_center - half_width, sigma_bounds[0], -np.inf, -np.inf]
    upper = [10 * abs(A_init) + 10, V_center + half_width, sigma_bounds[1],  np.inf,  np.inf]
    try:
        popt, pcov = curve_fit(gauss_lin, Vp, Np_, p0=p0, bounds=(lower, upper),
                               maxfev=30000)
    except Exception:
        return None, None, None, False
    perr = np.sqrt(np.diag(pcov))
    Vgrid = np.linspace(Vp.min(), Vp.max(), 4001)
    V_peak = float(Vgrid[int(np.argmax(gauss_lin(Vgrid, *popt)))])
    return popt, perr, V_peak, True


def fit_band_edge_gauss(V, norm,
                        coarse=COARSE_WINDOW,
                        rough_half_width=ROUGH_HALF_WIDTH,
                        tight_half_width=TIGHT_HALF_WIDTH,
                        sigma_bounds_rough=(SIGMA_LO, SIGMA_HI_ROUGH),
                        sigma_bounds_tight=(SIGMA_LO, SIGMA_HI_TIGHT),
                        reject_sigma_upper=False):
    """Explicit two-step Gaussian + linear-background peak fit.

    Step A: savgol-smooth (dI/dV)/(I/V) in `coarse` -> V_rough_argmax.
    Step B: rough Gaussian + linear fit in V_rough_argmax +/- rough_half_width.
             Returns V_peak_rough (more robust than the raw argmax,
             because the linear background is removed).
    Step C: tight Gaussian + linear fit in V_peak_rough +/- tight_half_width.
             The tight-fit V_peak and sigma are the reported values.

    Acceptance criterion is intentionally permissive: we want broad peaks
    (large sigma) at higher T to survive so we can plot sigma(T) and tell
    the band-smudging story. Rejection only triggers if the rough fit
    fails, the tight fit fails, the reported V_peak ends up outside the
    coarse window, or sigma is at the lower bound (sigma -> 0 is a
    runaway-narrow solution, not a real peak).

    Set `reject_sigma_upper=True` to additionally reject fits whose
    tight-fit sigma is pressed against the upper bound of
    sigma_bounds_tight. This recovers the strict 20 K-era behaviour where
    broad/shoulder-like peaks were discarded rather than reported with a
    clipped sigma. Pair it with a tighter sigma_bounds_tight (e.g.
    (0.03, 0.30) V at 20-30 K) to enforce a physical narrow-peak prior.
    """
    m = (V >= coarse[0]) & (V <= coarse[1])
    if m.sum() < SMOOTH_WIN + 2:
        return dict(ok=False, V_peak=np.nan, V_peak_err=np.nan,
                    sigma=np.nan, sigma_err=np.nan, popt=None, window=None)
    Vf, Nf = V[m], norm[m]
    wl = SMOOTH_WIN if SMOOTH_WIN <= len(Nf) and SMOOTH_WIN % 2 == 1 else max(5, (len(Nf) // 2) * 2 - 1)
    Ns = savgol_filter(Nf, window_length=wl, polyorder=SMOOTH_POLY)
    V_rough_arg = float(Vf[int(np.argmax(Ns))])

    # ---- Step B: rough Gaussian + linear in wide window ----
    m_rough = (Vf >= V_rough_arg - rough_half_width) & (Vf <= V_rough_arg + rough_half_width)
    Vp_r, Np_r = Vf[m_rough], Nf[m_rough]
    if Vp_r.size < 8:
        return dict(ok=False, V_peak=np.nan, V_peak_err=np.nan,
                    sigma=np.nan, sigma_err=np.nan, popt=None, window=None,
                    V_rough=V_rough_arg)
    popt_r, perr_r, V_peak_rough, ok_r = _fit_gauss_in_window(
        Vp_r, Np_r, V_rough_arg, rough_half_width, sigma_bounds_rough
    )
    if not ok_r:
        return dict(ok=False, V_peak=np.nan, V_peak_err=np.nan,
                    sigma=np.nan, sigma_err=np.nan, popt=None, window=None,
                    V_rough=V_rough_arg)

    # ---- Step C: tight Gaussian + linear in narrow window around V_peak_rough ----
    m_tight = (V >= V_peak_rough - tight_half_width) & (V <= V_peak_rough + tight_half_width)
    Vp_t, Np_t = V[m_tight], norm[m_tight]
    if Vp_t.size < 8:
        return dict(ok=False, V_peak=np.nan, V_peak_err=np.nan,
                    sigma=np.nan, sigma_err=np.nan, popt=None, window=None,
                    V_rough=V_rough_arg, V_peak_rough=V_peak_rough)
    popt_t, perr_t, V_peak_tight, ok_t = _fit_gauss_in_window(
        Vp_t, Np_t, V_peak_rough, tight_half_width, sigma_bounds_tight
    )
    if not ok_t:
        return dict(ok=False, V_peak=np.nan, V_peak_err=np.nan,
                    sigma=np.nan, sigma_err=np.nan, popt=None, window=None,
                    V_rough=V_rough_arg, V_peak_rough=V_peak_rough)

    A_t, V0_t, sigma_t, _, _ = popt_t
    eps = 1e-3
    # Permissive acceptance: reject only on runaway-narrow sigma and
    # V_peak outside the coarse band-edge window. Upper-bound rejection
    # is opt-in.
    bad_sigma_lo = sigma_t < sigma_bounds_tight[0] + eps
    bad_sigma_hi = reject_sigma_upper and sigma_t > sigma_bounds_tight[1] - eps
    bad_peak     = V_peak_tight < coarse[0] or V_peak_tight > coarse[1]
    ok = not (bad_sigma_lo or bad_sigma_hi or bad_peak)

    return dict(
        ok=ok,
        V_peak=V_peak_tight,
        V_peak_err=float(perr_t[1]),
        sigma=float(sigma_t),
        sigma_err=float(perr_t[2]),
        popt=popt_t,
        window=(float(Vp_t.min()), float(Vp_t.max())),
        V_rough=V_rough_arg,
        V_peak_rough=V_peak_rough,
        popt_rough=popt_r,
    )


def extract_peaks(df, **fit_kwargs):
    """Apply fit_band_edge_gauss to every row of an IV_gaussian dataframe.
    Returns a tidy dataframe with H, Phi_eV, Phi_err_eV, sigma, sigma_err, ok,
    and abs_H. Rows where ok is False are dropped.

    Any keyword arguments are forwarded to fit_band_edge_gauss, so callers
    can override sigma_bounds_tight, tight_half_width, reject_sigma_upper,
    etc. on a per-T basis without editing module defaults.
    """
    rows = []
    for _, r in df.iterrows():
        V, norm = normalised_didv(r['voltage_smooth'], r['current_smooth'])
        g = fit_band_edge_gauss(V, norm, **fit_kwargs)
        rows.append({
            'H':          r['H'],
            'Phi_eV':     g['V_peak'],
            'Phi_err_eV': g['V_peak_err'],
            'sigma':      g['sigma'],
            'sigma_err':  g['sigma_err'],
            'ok':         g['ok'],
        })
    out = pd.DataFrame(rows)
    out = out[out['ok']].reset_index(drop=True)
    out['abs_H'] = out['H'].abs()
    return out


def weighted_mean(values, errors):
    """Inverse-variance weighted mean with Birge-rescaled SEM."""
    v = np.asarray(values, float)
    e = np.asarray(errors, float)
    good = np.isfinite(v) & np.isfinite(e)
    v, e = v[good], e[good]
    if v.size == 0:
        return np.nan, np.nan, 0
    pos = e > 0
    if pos.any():
        e = np.where(pos, e, np.median(e[pos]))
    else:
        e = np.full_like(v, np.std(v, ddof=1) if v.size > 1 else 1.0)
    w = 1.0 / e**2
    mu       = float(np.sum(w * v) / np.sum(w))
    sigma_mu = float(1.0 / np.sqrt(np.sum(w)))
    if v.size > 1:
        chi2_red = float(np.sum(w * (v - mu)**2) / (v.size - 1))
        if chi2_red > 1.0:
            sigma_mu *= np.sqrt(chi2_red)
    return mu, sigma_mu, int(v.size)


def bin_weighted(df, key, round_decimals=2):
    """Group `df` by `df[key].round(round_decimals)` and weighted-average."""
    out = []
    for _, sub in df.groupby(df[key].round(round_decimals)):
        mu, sig, n = weighted_mean(sub['Phi_eV'], sub['Phi_err_eV'])
        out.append({key: float(sub[key].mean()),
                    'Phi_eV': mu, 'Phi_err_eV': sig, 'n': n})
    return pd.DataFrame(out).sort_values(key).reset_index(drop=True)


def canting_angle(H_z, H_sat):
    h = np.clip(np.abs(H_z) / H_sat, 0.0, 1.0)
    return 2.0 * np.arccos(h)


def sin2_half(H_z, H_sat):
    h = np.clip(np.abs(H_z) / H_sat, 0.0, 1.0)
    return 1.0 - h**2


__all__ = [
    'HALF_WIDTH', 'SIGMA_BOUNDS', 'COARSE_WINDOW',
    'ROUGH_HALF_WIDTH', 'TIGHT_HALF_WIDTH',
    'normalised_didv', 'gauss_lin', 'fit_band_edge_gauss',
    'extract_peaks', 'weighted_mean', 'bin_weighted',
    'canting_angle', 'sin2_half',
]
