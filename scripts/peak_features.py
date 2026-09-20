# ---
# description: |
#   Two-step Gaussian + linear-background peak feature extractor for the
#   Feenstra-normalised (dI/dV)/(I/V) curves. Step 1 (rough) fits inside
#   the user-supplied fit_region with the user-supplied peak_bounds, to
#   locate the peak robustly. Step 2 (fine) re-fits in a narrow window
#   +/- fine_half_width around the rough peak so the linear baseline is
#   determined locally and the reported sigma reflects the peak itself,
#   not the tails. Returns peak position, FWHM, height (above the linear
#   baseline) and their 1-sigma errors.
# entry_point: from scripts.peak_features import fit_peak_features, extract_peak_features
# dependencies:
#   - numpy, pandas, scipy
# input: |
#   fit_peak_features(V, norm, fit_region, peak_bounds, fwhm_bounds,
#   fine_half_width=0.18). extract_peak_features(df, fit_region, peak_bounds,
#   fwhm_bounds, fine_half_width=0.18) iterates over an IV dataframe whose
#   rows contain 'voltage_smooth','current_smooth','H'.
# process: |
#   Inner primitive _gauss_lin_fit: fit y = A*exp(-0.5((V-V0)/sigma)^2)+m V+c
#   with V0 inside peak_bounds, sigma in fwhm_bounds / FWHM_FACTOR, A >= 0.
#   Step 1: run the primitive on fit_region with peak_bounds -> rough peak.
#   Step 2: run the primitive on
#   [rough_peak - fine_half_width, rough_peak + fine_half_width] with
#   peak_bounds clipped into that window. Reported peak_pos_V is the
#   argmax of (Gaussian + linear baseline) on the fine grid (captures the
#   baseline-slope shift from V0). peak_pos_err is propagated through the
#   full 4x4 (A, V0, sigma, m) covariance via implicit-function partials
#   of f'(V_peak)=0, so the (typically strong) V0-m anticorrelation and
#   the sigma uncertainty both enter the reported peak error -- not just
#   V0_err diagonal.
# output: |
#   Per-curve dict with peak_pos_V, peak_pos_err, fwhm_V, fwhm_err,
#   height, height_err, popt, ok, plus rough_peak_V. The dataframe form
#   also exposes legacy aliases Phi_eV/Phi_err_eV/sigma/sigma_err so
#   downstream cells in the barrier/canting notebooks keep working.
# last_updated: 2026-05-25
# ---
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from scripts.barrier_canting_fit import gauss_lin, normalised_didv
from scripts.global_shared_fwhm import _full_peak_with_grad

FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))  # FWHM = FWHM_FACTOR * sigma
DEFAULT_FINE_HALF_WIDTH = 0.18


def _nan_result():
    return dict(
        ok=False,
        peak_pos_V=np.nan, peak_pos_err=np.nan,
        fwhm_V=np.nan,     fwhm_err=np.nan,
        height=np.nan,     height_err=np.nan,
        popt=None, perr=None, window=None, rough_peak_V=np.nan,
    )


def _gauss_lin_fit(V, norm, fit_region, peak_bounds, fwhm_bounds,
                   strict_peak_edge=True):
    """Single Gaussian + linear-baseline fit primitive.

    Returns (popt, perr, Vp, ok, V_peak_abs) where V_peak_abs is the
    argmax of (Gaussian + linear baseline) on a fine grid inside
    fit_region. Acceptance criterion is permissive: solver must
    converge, perr finite, and sigma must not be pinned at its lower
    bound (which is a runaway-narrow spurious solution). The upper
    sigma bound and V0 hitting the fine-window edge are NOT rejection
    triggers — at high T the peak is genuinely broad and the centre
    can drift. When strict_peak_edge=True (the rough step), V0 hitting
    the user-supplied peak_bounds boundary IS a reject (the optimiser
    ran into the user's prior).
    """
    R_lo, R_hi = fit_region
    V_lo = max(peak_bounds[0], R_lo)
    V_hi = min(peak_bounds[1], R_hi)
    if V_hi <= V_lo:
        return None, None, None, False, np.nan

    m = (V >= R_lo) & (V <= R_hi)
    Vp, Np_ = V[m], norm[m]
    if Vp.size < 8:
        return None, None, None, False, np.nan

    sigma_lo = fwhm_bounds[0] / FWHM_FACTOR
    sigma_hi = fwhm_bounds[1] / FWHM_FACTOR

    n_edge = max(3, len(Vp) // 6)
    m_init = ((Np_[-n_edge:].mean() - Np_[:n_edge].mean())
              / (Vp[-n_edge:].mean() - Vp[:n_edge].mean()))
    c_init = Np_[:n_edge].mean() - m_init * Vp[:n_edge].mean()
    V0_init = float(Vp[int(np.argmax(Np_ - (m_init * Vp + c_init)))])
    V0_init = float(np.clip(V0_init, V_lo, V_hi))
    sigma_init = float(np.clip(0.5 * (sigma_lo + sigma_hi), sigma_lo, sigma_hi))
    A_init = max(float(Np_.max() - (m_init * V0_init + c_init)), 1e-3)

    p0    = [A_init, V0_init, sigma_init, m_init, c_init]
    lower = [0.0,     V_lo,    sigma_lo, -np.inf, -np.inf]
    upper = [10 * abs(A_init) + 10, V_hi, sigma_hi, np.inf, np.inf]

    try:
        popt, pcov = curve_fit(gauss_lin, Vp, Np_, p0=p0,
                               bounds=(lower, upper), maxfev=30000)
    except Exception:
        return None, None, None, Vp, False, np.nan

    perr = np.sqrt(np.diag(pcov))
    _, V0, sigma, _, _ = popt
    _, V0_err, sigma_err, _, _ = perr

    Vgrid = np.linspace(R_lo, R_hi, 4001)
    V_peak_abs = float(Vgrid[int(np.argmax(gauss_lin(Vgrid, *popt)))])

    eps = 1e-6
    bad_sigma = sigma <= sigma_lo + eps  # runaway-narrow only
    bad_peak  = strict_peak_edge and (V0 <= V_lo + eps or V0 >= V_hi - eps)
    ok = bool(np.isfinite(V0_err) and np.isfinite(sigma_err)
              and not bad_sigma and not bad_peak)
    return popt, perr, pcov, Vp, ok, V_peak_abs


def fit_peak_features(V, norm, fit_region, peak_bounds, fwhm_bounds,
                      fine_half_width=DEFAULT_FINE_HALF_WIDTH):
    """Two-step Gaussian + linear-background peak fit.

    Parameters
    ----------
    V, norm : array-like
        Voltage and normalised dI/dV samples (already filtered).
    fit_region : (V_lo, V_hi)
        Absolute voltage window used in the rough (step 1) fit. Keeps the
        rough baseline determination local to the peak instead of using
        the whole IV curve.
    peak_bounds : (V0_lo, V0_hi)
        Bounds on the fitted peak position V0 in the rough fit. Clipped
        into fit_region.
    fwhm_bounds : (fwhm_lo, fwhm_hi)
        Absolute FWHM bounds in volts, applied in both rough and fine
        fits. Internally translated to sigma via FWHM = FWHM_FACTOR*sigma.
    fine_half_width : float, default 0.18 V
        Half-width of the fine-fit window centred on the rough peak.

    Returns a dict with peak_pos_V, peak_pos_err, fwhm_V, fwhm_err,
    height, height_err, ok (from the fine fit), and rough_peak_V.
    """
    V = np.asarray(V, float)
    norm = np.asarray(norm, float)

    # --- Step 1: rough fit on the user-supplied region ---
    popt_r, perr_r, _, _, ok_r, V_rough = _gauss_lin_fit(
        V, norm, fit_region, peak_bounds, fwhm_bounds,
        strict_peak_edge=True,
    )
    if not ok_r:
        return _nan_result()

    # --- Step 2: fine fit in a narrow window around the rough peak ---
    fine_region = (V_rough - fine_half_width, V_rough + fine_half_width)
    fine_peak_bounds = (V_rough - fine_half_width, V_rough + fine_half_width)
    popt_f, perr_f, pcov_f, Vp_f, ok_f, V_fine = _gauss_lin_fit(
        V, norm, fine_region, fine_peak_bounds, fwhm_bounds,
        strict_peak_edge=False,
    )
    if popt_f is None:
        out = _nan_result()
        out['rough_peak_V'] = float(V_rough)
        return out

    A, V0, sigma, m_lin, _ = popt_f
    A_err, V0_err, sigma_err, _, _ = perr_f

    # Propagate the full (A, V0, sigma, m) covariance into peak_pos_V using
    # the implicit-function partials of f'(V_peak)=0. The Gaussian centre
    # V0 is anti-correlated with the baseline slope m, so V0_err alone
    # systematically over-estimates the uncertainty on the actual peak
    # location. Sigma uncertainty enters here too.
    if pcov_f is not None and np.all(np.isfinite(pcov_f)):
        _, grad = _full_peak_with_grad(A, V0, sigma, m_lin,
                                       float(Vp_f.min()), float(Vp_f.max()))
        # popt_f layout is [A, V0, sigma, m, c] -> indices [0,1,2,3] for grad.
        C_sub = pcov_f[np.ix_([0, 1, 2, 3], [0, 1, 2, 3])]
        var_V_peak = float(grad @ C_sub @ grad)
        peak_pos_err = float(np.sqrt(max(var_V_peak, 0.0)))
    else:
        peak_pos_err = float(V0_err)

    return dict(
        ok=bool(ok_f),
        peak_pos_V=float(V_fine),
        peak_pos_err=peak_pos_err,
        fwhm_V=float(FWHM_FACTOR * sigma),
        fwhm_err=float(FWHM_FACTOR * sigma_err),
        height=float(A),
        height_err=float(A_err),
        popt=popt_f,
        perr=perr_f,
        window=(float(Vp_f.min()), float(Vp_f.max())),
        rough_peak_V=float(V_rough),
    )


def extract_peak_features(df, fit_region, peak_bounds, fwhm_bounds,
                          fine_half_width=DEFAULT_FINE_HALF_WIDTH,
                          voltage_col='voltage_smooth',
                          current_col='current_smooth'):
    """Apply the two-step fit_peak_features to every row of an IV dataframe.

    Returns a tidy dataframe with the new feature columns
    (peak_pos_V, peak_pos_err, fwhm_V, fwhm_err, height, height_err,
    rough_peak_V) and legacy aliases (Phi_eV, Phi_err_eV, sigma, sigma_err)
    so the downstream barrier/canting analysis cells keep working
    unchanged. Rows whose fit failed are dropped.
    """
    rows = []
    for _, r in df.iterrows():
        V, norm = normalised_didv(r[voltage_col], r[current_col])
        g = fit_peak_features(V, norm, fit_region=fit_region,
                              peak_bounds=peak_bounds,
                              fwhm_bounds=fwhm_bounds,
                              fine_half_width=fine_half_width)
        sigma_val = (g['fwhm_V'] / FWHM_FACTOR) if np.isfinite(g['fwhm_V']) else np.nan
        sigma_err = (g['fwhm_err'] / FWHM_FACTOR) if np.isfinite(g['fwhm_err']) else np.nan
        rows.append({
            'H':             r['H'],
            'peak_pos_V':    g['peak_pos_V'],
            'peak_pos_err':  g['peak_pos_err'],
            'fwhm_V':        g['fwhm_V'],
            'fwhm_err':      g['fwhm_err'],
            'height':        g['height'],
            'height_err':    g['height_err'],
            'rough_peak_V':  g['rough_peak_V'],
            'Phi_eV':        g['peak_pos_V'],
            'Phi_err_eV':    g['peak_pos_err'],
            'sigma':         sigma_val,
            'sigma_err':     sigma_err,
            'ok':            g['ok'],
        })
    out = pd.DataFrame(rows)
    out = out[out['ok']].reset_index(drop=True)
    out['abs_H'] = out['H'].abs()
    return out


__all__ = ['FWHM_FACTOR', 'DEFAULT_FINE_HALF_WIDTH',
           'fit_peak_features', 'extract_peak_features']
