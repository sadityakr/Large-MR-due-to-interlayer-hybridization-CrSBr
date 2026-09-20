"""Empirical point-to-point uncertainty from the scatter of a series about its trend.

The errors carried by the MR summary CSVs are standard errors of the mean over the
field points inside each magnetic state. Those field points are not independent
repeats: within one state the current still drifts smoothly with field (lag-1
autocorrelation 0.9 or higher), so dividing the spread by the square root of the
sample size understates the real uncertainty. The consequence is visible in the
figures: consecutive points scatter by several times their own error bar. This
module measures that scatter directly from the plotted series and uses it as a
floor on the quoted error, so no point claims a precision the data does not show.

Input: the abscissa and ordinate of one plotted series (temperature and peak MR,
or bias and MR ratio), together with the propagated one-sigma error to be floored.
Optionally a range of the abscissa over which the trend is smooth enough for the
estimator to be valid.

Process: the second difference ``y[i-1] - 2 y[i] + y[i+1]`` cancels any locally
linear trend, so on an evenly spaced abscissa its variance is six times the
per-point variance of independent noise. Triples whose two spacings differ are
skipped, which keeps gaps in the abscissa (the current-noise cut around zero bias)
from being read as noise. The resulting single number per series is then applied
as an elementwise floor to the propagated error.

Output: a scalar one-sigma scatter, or an error array in which every entry is the
larger of the propagated error and that scatter.
"""

from __future__ import annotations

import numpy as np

# Temperature range over which max(MR) and the peak bias vary smoothly enough for a
# three-point locally linear model to hold. Below it the current-noise cut removes
# most of the bias sweep; above about 100 K the MR collapses towards the zero-bias
# cusp steeply enough that genuine curvature would be counted as noise, which would
# overstate the uncertainty instead of measuring it.
SMOOTH_T_RANGE_K = (20.0, 90.0)

# Variance of a second difference of independent points, in units of the per-point
# variance: (+1, -2, +1) has squared norm 1 + 4 + 1 = 6.
_SECOND_DIFFERENCE_VARIANCE = 6.0


def trend_scatter(x, y, x_range=None, spacing_rtol=0.05):
    """One-sigma point-to-point scatter of ``y`` about a locally linear trend in ``x``.

    Parameters
    ----------
    x, y : array_like
        Abscissa and ordinate of one plotted series.
    x_range : tuple of float, optional
        ``(x_min, x_max)`` restricting the estimate to the part of the series
        where the trend is smooth. Points outside it are excluded from the
        estimate, but the returned scatter still applies to the whole series
        because the measurement conditions do not change.
    spacing_rtol : float
        Relative tolerance on the two spacings of a triple. Triples that straddle
        a gap in the abscissa are skipped rather than counted as noise.

    Returns
    -------
    float
        One-sigma scatter in the units of ``y``, or ``nan`` if fewer than three
        usable consecutive points remain.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    good = np.isfinite(x) & np.isfinite(y)
    if x_range is not None:
        good &= (x >= x_range[0]) & (x <= x_range[1])
    x, y = x[good], y[good]

    order = np.argsort(x)
    x, y = x[order], y[order]
    if len(x) < 3:
        return np.nan

    left = x[1:-1] - x[:-2]
    right = x[2:] - x[1:-1]
    even = np.abs(right - left) <= spacing_rtol * np.maximum(left, right)
    if not even.any():
        return np.nan

    second = (y[:-2] - 2.0 * y[1:-1] + y[2:])[even]
    return float(np.sqrt(np.mean(second ** 2) / _SECOND_DIFFERENCE_VARIANCE))


def with_trend_floor(error, x, y, x_range=None):
    """Raise a propagated error to the measured scatter wherever it falls below it.

    The elementwise maximum is deliberately conservative in both directions: it
    never shrinks a propagated error that is already large (the plateau fallback
    on the peak bias above 100 K), and never lets one sit below the scatter the
    series actually shows.

    Returns
    -------
    tuple
        ``(error_array, scatter)`` so the caller can print the scatter that was
        applied alongside the figure.
    """
    error = np.asarray(error, dtype=float)
    scatter = trend_scatter(x, y, x_range=x_range)
    if not np.isfinite(scatter):
        return error, scatter
    return np.maximum(error, scatter), scatter
