"""Uncertainty on the position and height of the MR-vs-bias maximum.

The MR summary CSVs carry a statistical error on the ratio itself (``TMR_Error``)
but none on the *location* of the maximum, because that location is a derived
quantity rather than a measured one. This module supplies both: the peak height
with its measured error, and the peak bias with an error obtained from a local
inverse-variance-weighted parabola fit around the maximum. It is used by the
tunneling-regime notebooks so that every peak-versus-temperature point carries
an error bar.

Input: three equal-length arrays for one temperature and one crystal axis, and
one bias branch: bias voltage (V), MR ratio (dimensionless, I_apar/I_par), and
the ratio's one-sigma error. Points must already be filtered for the current
noise floor.

Process: locate the discrete argmax, fit ``y = a (V - V0)^2 + b (V - V0) + c``
to the points within ``half_window`` volts of it with weights ``1 / sigma^2``,
and propagate the fit covariance to the vertex ``V0 - b / (2a)``. The vertex
error is scaled by ``sqrt(chi2_red)`` when the reduced chi-square exceeds one,
so that curve shape not captured by a parabola inflates the error instead of
being ignored. The fit is accepted only if the curvature is negative and the
vertex falls inside the fit window; otherwise the maximum is not an interior
extremum (above roughly 100 K it collapses onto the zero-bias cusp) and the
routine falls back to the half-width of the bias interval over which the MR
stays within one sigma of its maximum. Both estimates are floored at half the
bias step, since no estimator can localise a peak better than the sampling.

Output: a dict with the peak bias and its error, the peak ratio and its error,
the reduced chi-square, the number of points in the fit, which estimator was
used (``"parabola"`` or ``"plateau"``), and a ``resolved`` flag that is False
whenever the fallback had to be used. The peak height is reported twice:
``max_ratio`` is the highest measured point, and ``fit_ratio`` is the vertex of
the same parabola. The discrete maximum is biased upward, since the point that
scatters high is the one selected, so ``fit_ratio`` is the better estimate of
the peak height wherever the fit is accepted.
"""

from __future__ import annotations

import numpy as np

# Half-width of the parabola fit window in volts. Chosen from the measured peak
# shape: the MR maximum is several hundred mV wide, the bias step is ~10 mV, so
# +-0.10 V gives ~19 points per fit. Narrower windows are noise-dominated (the
# fitted curvature changes sign); wider ones leave the parabolic region and the
# vertex drifts systematically.
DEFAULT_HALF_WINDOW_V = 0.10

# Fewest points required before a three-parameter parabola is attempted.
MIN_FIT_POINTS = 7


def _bias_step(V):
    """Median spacing of the bias grid, used as the resolution floor."""
    if len(V) < 2:
        return np.nan
    return float(np.median(np.diff(V)))


def _plateau_half_width(V, y, sy, k):
    """Half-width of the bias interval where the MR stays within 1 sigma of its max.

    This is the fallback estimator: it makes no assumption about the peak shape
    and simply reports over what bias range the data cannot distinguish the MR
    from its maximum value.
    """
    threshold = y[k] - sy[k]
    inside = y >= threshold
    lo = hi = k
    while lo > 0 and inside[lo - 1]:
        lo -= 1
    while hi < len(V) - 1 and inside[hi + 1]:
        hi += 1
    return 0.5 * float(V[hi] - V[lo])


def peak_voltage_with_error(voltage, ratio, ratio_error,
                            half_window=DEFAULT_HALF_WINDOW_V,
                            min_points=MIN_FIT_POINTS):
    """Locate the MR maximum in bias and attach an uncertainty to both coordinates.

    Parameters
    ----------
    voltage, ratio, ratio_error : array_like
        Bias voltage (V), MR ratio, and the ratio's one-sigma error, for one
        temperature, one crystal axis and one bias branch.
    half_window : float
        Half-width in volts of the parabola fit window around the discrete argmax.
    min_points : int
        Minimum number of points inside the window for the fit to be attempted.

    Returns
    -------
    dict or None
        None if fewer than three usable points are supplied. Otherwise keys
        ``peak_voltage``, ``peak_voltage_error``, ``max_ratio``,
        ``max_ratio_error``, ``chi2_red``, ``n_fit``, ``method``, ``resolved``.
    """
    V = np.asarray(voltage, dtype=float)
    y = np.asarray(ratio, dtype=float)
    sy = np.asarray(ratio_error, dtype=float)

    good = np.isfinite(V) & np.isfinite(y) & np.isfinite(sy) & (sy > 0)
    V, y, sy = V[good], y[good], sy[good]
    if len(V) < 3:
        return None

    order = np.argsort(V)
    V, y, sy = V[order], y[order], sy[order]

    k = int(np.argmax(y))
    step = _bias_step(V)
    floor = 0.5 * step if np.isfinite(step) else 0.0

    result = {
        'peak_voltage': float(V[k]),
        'peak_voltage_error': np.nan,
        'max_ratio': float(y[k]),
        'max_ratio_error': float(sy[k]),
        # Peak height read off the fitted parabola rather than from the single
        # highest point. The discrete maximum of a noisy curve is biased upward,
        # because whichever point happens to scatter high is the one selected;
        # the vertex of a fit to the whole peak is not. NaN when the parabola is
        # rejected, so callers can fall back to max_ratio.
        'fit_ratio': np.nan,
        'fit_ratio_error': np.nan,
        'chi2_red': np.nan,
        'n_fit': 0,
        'method': 'plateau',
        'resolved': False,
    }

    sel = np.abs(V - V[k]) <= half_window
    n_fit = int(sel.sum())
    result['n_fit'] = n_fit

    if n_fit >= min_points:
        V0 = V[k]
        x = V[sel] - V0
        yw = y[sel]
        w = 1.0 / sy[sel] ** 2
        design = np.vstack([x ** 2, x, np.ones_like(x)]).T
        weighted = design.T * w
        try:
            cov = np.linalg.inv(weighted @ design)
        except np.linalg.LinAlgError:
            cov = None

        if cov is not None:
            params = cov @ (weighted @ yw)
            a, b, c = float(params[0]), float(params[1]), float(params[2])
            residual = yw - design @ params
            dof = max(n_fit - 3, 1)
            chi2_red = float(np.sum(w * residual ** 2) / dof)
            result['chi2_red'] = chi2_red

            # Three conditions have to hold before the vertex is believable:
            # downward curvature, a vertex that lies inside the bias range the
            # parabola was actually fitted to (otherwise it is extrapolating and
            # the "peak" is really an edge of the measured range), and an error
            # no larger than the fit window itself (otherwise the fit has not
            # localised the peak at all).
            if a < 0:
                vertex = V0 - b / (2.0 * a)
                # Gradient of the vertex position with respect to (a, b, c).
                grad = np.array([b / (2.0 * a ** 2), -1.0 / (2.0 * a), 0.0])
                var = float(grad @ cov @ grad)
                sigma = np.sqrt(max(var, 0.0))
                # Inflate by sqrt(chi2_red) when the parabola does not fully
                # describe the data: the standard way to fold model mismatch
                # into a covariance-derived error bar.
                if chi2_red > 1.0:
                    sigma *= np.sqrt(chi2_red)
                inside = float(V[sel].min()) <= vertex <= float(V[sel].max())
                if inside and sigma <= half_window:
                    result['peak_voltage'] = float(vertex)
                    result['peak_voltage_error'] = float(max(sigma, floor))

                    # Vertex height of the same parabola, y(x*) = c - b^2 / (4a),
                    # with its error from the same covariance. Averaging over the
                    # whole peak makes this error smaller than the single-point
                    # one, and removes the upward bias of the discrete maximum.
                    height = c - b ** 2 / (4.0 * a)
                    grad_h = np.array([b ** 2 / (4.0 * a ** 2), -b / (2.0 * a), 1.0])
                    sigma_h = np.sqrt(max(float(grad_h @ cov @ grad_h), 0.0))
                    if chi2_red > 1.0:
                        sigma_h *= np.sqrt(chi2_red)
                    result['fit_ratio'] = float(height)
                    result['fit_ratio_error'] = float(sigma_h)

                    result['method'] = 'parabola'
                    result['resolved'] = True
                    return result

    # Fallback: the maximum is not an interior extremum of a parabolic peak.
    result['peak_voltage_error'] = float(max(_plateau_half_width(V, y, sy, k), floor))
    return result
