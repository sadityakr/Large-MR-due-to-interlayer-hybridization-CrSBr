# ---
# description: |
#   Global shared-sigma (shared-FWHM) Gaussian + linear-background fit
#   across many dI/dV curves from an H-sweep. One sigma is fitted jointly
#   to all curves, while amplitude, peak centre, and linear background
#   (slope + intercept) remain per-curve. The reported V_peak is the
#   argmax of the full model (Gaussian + linear baseline), not just the
#   Gaussian centre V0 -- a non-zero baseline slope shifts the true peak
#   away from V0. Its error is propagated from the full 4-parameter
#   covariance block (A, V0, shared sigma, m), so the shared-sigma
#   uncertainty correctly enters the per-curve barrier-height error. Also
#   provides a measured FWHM that does not assume a Gaussian shape:
#   half-max crossings on the baseline-subtracted data, with crossing-
#   uncertainty propagated from the local data noise.
# entry_point: from scripts.global_shared_fwhm import (
#   prepare_curves, fit_shared_sigma_global, measured_fwhm,
#   apply_manual_overrides)
# dependencies:
#   - numpy, pandas, scipy
# input: |
#   prepare_curves: an IV_gaussian dataframe (cols: H, voltage_smooth,
#   current_smooth) plus a tight_half_width and rough-fit kwargs forwarded
#   to scripts.barrier_canting_fit.fit_band_edge_gauss. fit_shared_sigma_global:
#   the curve list. measured_fwhm: per-curve V, N arrays + the per-curve
#   linear baseline (m, c) from the global fit.
# process: |
#   prepare_curves runs the existing two-step rough fit to get V_peak_rough
#   per curve and clips data to V_peak_rough +/- tight_half_width. The
#   global fit packs parameters as [sigma, (A_i, V0_i, m_i, c_i) for each
#   curve] and minimises stacked residuals with scipy.optimize.least_squares.
#   The per-curve V_peak is the argmax of A*exp(-((V-V0)/sigma)^2/2)+m*V+c
#   on a fine grid in the curve's tight window. The error on V_peak is
#   computed by implicit differentiation of f'(V_peak)=0 with respect to
#   (A, V0, sigma, m), then propagated through the 4x4 covariance block
#   that includes cross-terms with the shared sigma. measured_fwhm
#   subtracts the linear baseline, locates the peak by argmax on a
#   savgol-smoothed signal, linearly interpolates the left/right crossings
#   at A/2, and reports the absolute half-max width with a 1-sigma error
#   derived from local point-to-point noise.
# output: |
#   fit_shared_sigma_global returns a dict with: sigma, sigma_err,
#   per-curve dataframe (H, A, V_peak, V_peak_err, V0_gauss, V0_gauss_err,
#   slope, slope_err, intercept, intercept_err, chi2, n_points). V_peak is
#   the full-model argmax with covariance-propagated error; V0_gauss is
#   the fitted Gaussian centre with its diagonal-cov error (kept so that
#   the per-curve Gaussian curve can still be drawn for spot-checks).
#   measured_fwhm returns (fwhm_V, fwhm_err, V_left, V_right, height) per
#   curve.
# last_updated: 2026-05-25
# ---
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

from scripts.barrier_canting_fit import (
    normalised_didv, gauss_lin, fit_band_edge_gauss,
    TIGHT_HALF_WIDTH, SIGMA_LO, SIGMA_HI_TIGHT,
)


def prepare_curves(df, tight_half_width=TIGHT_HALF_WIDTH, **rough_kwargs):
    """Build per-curve (V, N) windows around V_peak_rough.

    Returns a list of dicts: H, V (windowed), N (windowed), V_peak_rough,
    voltage_full, norm_full, plus the original index in df.
    """
    curves = []
    for idx, r in df.iterrows():
        V, N = normalised_didv(r['voltage_smooth'], r['current_smooth'])
        g = fit_band_edge_gauss(V, N, **rough_kwargs)
        V_rough = g.get('V_peak_rough', np.nan)
        if not np.isfinite(V_rough):
            continue
        m = (V >= V_rough - tight_half_width) & (V <= V_rough + tight_half_width)
        if m.sum() < 10:
            continue
        curves.append(dict(
            df_index=idx,
            H=float(r['H']),
            V=V[m], N=N[m],
            V_peak_rough=float(V_rough),
            voltage_full=V, norm_full=N,
        ))
    return curves


def _pack(sigma, per):
    return np.concatenate([[sigma], per.ravel()])


def _unpack(params, n):
    sigma = params[0]
    per = params[1:].reshape(n, 4)
    return sigma, per


def _residuals(params, curves):
    sigma, per = _unpack(params, len(curves))
    out = []
    for (A, V0, m, c0), c in zip(per, curves):
        model = A * np.exp(-0.5 * ((c['V'] - V0) / sigma) ** 2) + m * c['V'] + c0
        out.append(c['N'] - model)
    return np.concatenate(out)


def _full_peak_with_grad(A, V0, sigma, m, V_min, V_max, n_grid=8001):
    """Argmax of f(V) = A*exp(-((V-V0)/sigma)^2/2) + m*V + c on a fine
    grid, with the gradient [dV_peak/dA, dV_peak/dV0, dV_peak/dsigma,
    dV_peak/dm] obtained by implicit differentiation of f'(V_peak)=0.

    The intercept c does not enter the peak position. Returns
    (V_peak, gradient_vector).
    """
    Vg = np.linspace(V_min, V_max, n_grid)
    f = A * np.exp(-0.5 * ((Vg - V0) / sigma) ** 2) + m * Vg
    V_peak = float(Vg[int(np.argmax(f))])

    u = (V_peak - V0) / sigma
    E = float(np.exp(-0.5 * u * u))
    # G(V) := f'(V) ; we have G(V_peak) = 0. dG/dV at V_peak:
    dG_dV = -A / (sigma * sigma) * (1.0 - u * u) * E
    if not np.isfinite(dG_dV) or abs(dG_dV) < 1e-12 or abs(1.0 - u * u) < 1e-9:
        # Degenerate (inflection / very flat); fall back to V0-shift only.
        return V_peak, np.array([0.0, 1.0, 0.0, 0.0])

    # Implicit-function partials: dV*/dtheta = -(dG/dtheta) / (dG/dV).
    dV_dA  = -(-u / sigma * E) / dG_dV
    dV_dV0 = 1.0
    dV_ds  = -(-A * u / (sigma * sigma) * (u * u - 2.0) * E) / dG_dV
    dV_dm  = -1.0 / dG_dV
    return V_peak, np.array([dV_dA, dV_dV0, dV_ds, dV_dm])


def fit_shared_sigma_global(curves, sigma_init=None,
                            sigma_bounds=(SIGMA_LO, SIGMA_HI_TIGHT),
                            fixed=None):
    """Joint least-squares with one shared sigma across all curves.

    Parameters
    ----------
    curves : list from prepare_curves.
    sigma_init : optional initial sigma; defaults to 0.06 V.
    sigma_bounds : (low, high) bounds on shared sigma.
    fixed : optional dict mapping df_index -> dict of pinned per-curve
        params (any subset of {A, V0, m, c}). Pinned params are removed
        from the optimisation by clamping lower==upper.

    Returns dict with sigma, sigma_err, df (per-curve table), residual,
    cost, success.
    """
    n = len(curves)
    fixed = fixed or {}

    # Build initial per-curve params from a quick per-curve fit.
    p0_per = np.zeros((n, 4))
    lower_per = np.full((n, 4), -np.inf)
    upper_per = np.full((n, 4),  np.inf)
    for i, c in enumerate(curves):
        Vc, Nc = c['V'], c['N']
        n_edge = max(3, len(Vc) // 6)
        m_init = (Nc[-n_edge:].mean() - Nc[:n_edge].mean()) / (Vc[-n_edge:].mean() - Vc[:n_edge].mean())
        c_init = Nc[:n_edge].mean() - m_init * Vc[:n_edge].mean()
        A_init = max(float(Nc.max() - (m_init * c['V_peak_rough'] + c_init)), 1e-3)
        p0_per[i] = [A_init, c['V_peak_rough'], m_init, c_init]
        lower_per[i, 0] = 0.0
        lower_per[i, 1] = c['V_peak_rough'] - 0.5 * (c['V'].max() - c['V'].min())
        upper_per[i, 1] = c['V_peak_rough'] + 0.5 * (c['V'].max() - c['V'].min())

        # Manual override: pin selected params to user-supplied values.
        ov = fixed.get(c['df_index'])
        if ov:
            for key, j in (('A', 0), ('V0', 1), ('m', 2), ('c', 3)):
                if key in ov and np.isfinite(ov[key]):
                    p0_per[i, j] = float(ov[key])
                    lower_per[i, j] = float(ov[key]) - 1e-9
                    upper_per[i, j] = float(ov[key]) + 1e-9

    sigma0 = float(sigma_init) if sigma_init is not None else 0.06
    sigma0 = float(np.clip(sigma0, sigma_bounds[0] + 1e-6, sigma_bounds[1] - 1e-6))

    p0 = _pack(sigma0, p0_per)
    lower = _pack(sigma_bounds[0], lower_per)
    upper = _pack(sigma_bounds[1], upper_per)

    res = least_squares(_residuals, p0, bounds=(lower, upper),
                        args=(curves,), max_nfev=50000)

    sigma_fit, per_fit = _unpack(res.x, n)

    # Covariance from Jacobian at solution (Gauss-Newton approx).
    try:
        J = res.jac
        n_data = J.shape[0]
        n_par  = J.shape[1]
        resid_var = 2.0 * res.cost / max(n_data - n_par, 1)
        JTJ = J.T @ J
        cov = resid_var * np.linalg.pinv(JTJ)
        diag = np.diag(cov)
        sigma_err = float(np.sqrt(max(diag[0], 0.0)))
        per_err = np.sqrt(np.clip(diag[1:].reshape(n, 4), 0.0, None))
        cov_ok = True
    except Exception:
        sigma_err = np.nan
        per_err = np.full_like(per_fit, np.nan)
        cov = None
        cov_ok = False

    rows = []
    resid = res.fun
    offset = 0
    for i, c in enumerate(curves):
        npts = len(c['V'])
        chi2 = float(np.sum(resid[offset:offset + npts] ** 2))
        offset += npts
        A, V0, m, c0 = per_fit[i]
        Ae, V0e, me, c0e = per_err[i]

        # Full-model peak (Gaussian + linear baseline), argmax in the
        # curve's tight window. The intercept c0 does not affect the peak.
        V_peak_full, grad = _full_peak_with_grad(
            A, V0, sigma_fit, m, float(c['V'].min()), float(c['V'].max())
        )

        # Propagate (A_i, V0_i, sigma_shared, m_i) covariance into V_peak.
        # Param indices in the full vector: sigma=0; per-curve (A, V0, m, c)
        # at 1+4i, 2+4i, 3+4i, 4+4i.
        if cov_ok:
            iA, iV0, iM = 1 + 4 * i, 2 + 4 * i, 3 + 4 * i
            idx = np.array([iA, iV0, 0, iM])
            C_sub = cov[np.ix_(idx, idx)]
            var_V_peak = float(grad @ C_sub @ grad)
            V_peak_full_err = float(np.sqrt(max(var_V_peak, 0.0)))
        else:
            V_peak_full_err = np.nan

        rows.append(dict(
            df_index=c['df_index'],
            H=c['H'],
            A=A, A_err=Ae,
            V_peak=V_peak_full, V_peak_err=V_peak_full_err,
            V0_gauss=V0, V0_gauss_err=V0e,
            slope=m, slope_err=me,
            intercept=c0, intercept_err=c0e,
            chi2=chi2, n_points=npts,
            V_peak_rough=c['V_peak_rough'],
        ))
    df_out = pd.DataFrame(rows)
    df_out['abs_H'] = df_out['H'].abs()

    return dict(
        sigma=float(sigma_fit), sigma_err=sigma_err,
        df=df_out,
        success=bool(res.success),
        cost=float(res.cost),
        message=str(res.message),
    )


def measured_fwhm(V, N, slope, intercept, smooth_window=11, smooth_poly=3):
    """Fit-free FWHM from baseline-subtracted half-max crossings.

    Subtracts the linear background `slope * V + intercept`, finds the
    peak by argmax on a savgol-smoothed signal, locates the left and
    right voltages where the signal crosses height/2 by linear
    interpolation between adjacent samples. The error on the FWHM is
    propagated from the local point-to-point noise via
    delta_V_cross = sqrt(2) * sigma_noise / |dN/dV| at the crossing.
    """
    V = np.asarray(V, float)
    N = np.asarray(N, float)
    if V.size < 7:
        return dict(fwhm_V=np.nan, fwhm_err=np.nan,
                    V_left=np.nan, V_right=np.nan, height=np.nan,
                    sigma_noise=np.nan)
    S = N - (slope * V + intercept)
    wl = smooth_window if smooth_window <= len(S) and smooth_window % 2 == 1 else max(5, (len(S) // 2) * 2 - 1)
    S_smooth = savgol_filter(S, window_length=wl, polyorder=min(smooth_poly, wl - 1))
    i_max = int(np.argmax(S_smooth))
    height = float(S_smooth[i_max])
    if height <= 0 or i_max in (0, len(S) - 1):
        return dict(fwhm_V=np.nan, fwhm_err=np.nan,
                    V_left=np.nan, V_right=np.nan, height=height,
                    sigma_noise=np.nan)
    half = 0.5 * height

    # Local noise from data minus smoothed (high-frequency residual).
    sigma_noise = float(np.std(S - S_smooth, ddof=1))

    def cross(i_from, step):
        i = i_from
        while 0 <= i + step < len(S_smooth):
            if (S_smooth[i] - half) * (S_smooth[i + step] - half) <= 0:
                # Linear interpolation between sample i and i+step.
                x0, x1 = V[i], V[i + step]
                y0, y1 = S_smooth[i], S_smooth[i + step]
                if y1 == y0:
                    return float(0.5 * (x0 + x1)), abs(x1 - x0)
                t = (half - y0) / (y1 - y0)
                Vc = x0 + t * (x1 - x0)
                dydV = (y1 - y0) / (x1 - x0)
                err = float(np.sqrt(2.0) * sigma_noise / abs(dydV)) if dydV != 0 else np.nan
                return float(Vc), float(err)
            i += step
        return np.nan, np.nan

    V_left,  err_left  = cross(i_max, -1)
    V_right, err_right = cross(i_max, +1)
    if not (np.isfinite(V_left) and np.isfinite(V_right)):
        return dict(fwhm_V=np.nan, fwhm_err=np.nan,
                    V_left=V_left, V_right=V_right, height=height,
                    sigma_noise=sigma_noise)
    fwhm = V_right - V_left
    fwhm_err = float(np.sqrt(err_left ** 2 + err_right ** 2))
    return dict(fwhm_V=float(fwhm), fwhm_err=fwhm_err,
                V_left=V_left, V_right=V_right, height=height,
                sigma_noise=sigma_noise)


def add_measured_fwhm(per_df, curves, fit_df, search_half_width=0.45):
    """For each row of fit_df, compute the measured (fit-free) FWHM using
    the full normalised curve restricted to V_peak +/- search_half_width,
    with the global-fit linear baseline (slope, intercept) subtracted.
    search_half_width must be wider than the expected FWHM/2 so the
    half-max crossings live inside the search window.
    """
    by_idx = {c['df_index']: c for c in curves}
    out = fit_df.copy()
    cols = dict(fwhm_meas_V=[], fwhm_meas_err=[],
                V_left_meas=[], V_right_meas=[],
                height_meas=[], sigma_noise=[])
    for _, row in out.iterrows():
        c = by_idx[row['df_index']]
        Vf, Nf = c['voltage_full'], c['norm_full']
        mask = (Vf >= row['V_peak'] - search_half_width) & (Vf <= row['V_peak'] + search_half_width)
        if mask.sum() < 10:
            m = measured_fwhm(c['V'], c['N'], row['slope'], row['intercept'])
        else:
            m = measured_fwhm(Vf[mask], Nf[mask], row['slope'], row['intercept'])
        cols['fwhm_meas_V'].append(m['fwhm_V'])
        cols['fwhm_meas_err'].append(m['fwhm_err'])
        cols['V_left_meas'].append(m['V_left'])
        cols['V_right_meas'].append(m['V_right'])
        cols['height_meas'].append(m['height'])
        cols['sigma_noise'].append(m['sigma_noise'])
    for k, v in cols.items():
        out[k] = v
    return out


__all__ = [
    'prepare_curves', 'fit_shared_sigma_global',
    'measured_fwhm', 'add_measured_fwhm',
    '_full_peak_with_grad',
]
