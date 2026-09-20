"""Propagate the measured current error through the I(V) transforms used for plotting.

The MR summary CSVs carry a one-sigma error on every current point, but the
quantities actually plotted are transforms of the current: the Fowler-Nordheim
ordinate, the numerical derivative dI/dV, its Feenstra-normalised form
(dI/dV)/(I/V), and the second derivative. This module turns the per-point
current errors into error bars on those transforms, so that the derivative
panels can be shown with the same honesty as the raw data.

Input: the bias grid (V) and the current with its one-sigma error, as plain
arrays for one temperature, one crystal axis and one magnetic state.

Process: numerical differentiation with ``np.gradient`` is a linear operation,
so its Jacobian is obtained exactly by differentiating the identity matrix on
the same grid. Errors then propagate as ``sigma_out = sqrt((J**2) @ sigma_in**2)``,
which is exact for the linear cases and correct to first order for the
normalised ratio. Because ``np.gradient`` uses a central difference, the
derivative at a point does not involve the current at that same point; the
Jacobian handles that correlation structure automatically rather than assuming
it away.

Output: arrays of the transformed quantity and its one-sigma error, ready to
pass to ``ax.errorbar``, or to ``fill_sigma_band`` for the dense curves where a
shaded band reads better than a cap on every point.
"""

from __future__ import annotations

import numpy as np


def fill_sigma_band(ax, voltage, value, error, **kwargs):
    """Shade ``value +- error`` against bias, drawn separately per bias polarity.

    Every I(V) transform plotted here has a hole around zero bias, from either
    the current-noise cut or an explicit ``|V| > 0.05`` mask. A single
    ``fill_between`` call would bridge that hole with a band across a region
    where nothing was measured, so the two polarities are shaded separately.
    """
    V = np.asarray(voltage, dtype=float)
    y = np.asarray(value, dtype=float)
    e = np.asarray(error, dtype=float)
    for side in (V < 0, V > 0):
        if side.sum() > 1:
            ax.fill_between(V[side], (y - e)[side], (y + e)[side],
                            linewidth=0, **kwargs)


def gradient_jacobian(V):
    """Jacobian of ``np.gradient(y, V)`` with respect to ``y``.

    ``np.gradient`` is linear in its data argument, so applying it column-wise
    to an identity matrix reconstructs the exact matrix ``G`` for which
    ``np.gradient(y, V) == G @ y``. This works for any grid spacing, uniform or
    not, and stays in step with whatever stencil numpy uses.
    """
    V = np.asarray(V, dtype=float)
    n = len(V)
    return np.gradient(np.eye(n), V, axis=0)


def _propagate(jacobian, sigma):
    """One-sigma error of ``jacobian @ x`` given independent errors on ``x``."""
    return np.sqrt((jacobian ** 2) @ np.asarray(sigma, dtype=float) ** 2)


def fn_ordinate_error(current, current_error):
    """One-sigma error of the Fowler-Nordheim ordinate ``ln(|I| / V^2)``.

    The bias enters only through a term with no uncertainty, so the whole error
    is the relative current error ``sigma_I / |I|``.
    """
    I = np.asarray(current, dtype=float)
    sI = np.asarray(current_error, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.abs(sI / I)


def didv_with_error(V, current, current_error):
    """Numerical dI/dV and its one-sigma error."""
    V = np.asarray(V, dtype=float)
    I = np.asarray(current, dtype=float)
    sI = np.asarray(current_error, dtype=float)
    G = gradient_jacobian(V)
    return G @ I, _propagate(G, sI)


def d2idv2_with_error(V, current, current_error):
    """Numerical d2I/dV2 and its one-sigma error.

    Differentiating twice squares the ``1 / spacing`` amplification, so the
    error bars here are large by construction. That is the point: it shows how
    much of the second-derivative structure is noise.
    """
    V = np.asarray(V, dtype=float)
    I = np.asarray(current, dtype=float)
    sI = np.asarray(current_error, dtype=float)
    G = gradient_jacobian(V)
    G2 = G @ G
    return G2 @ I, _propagate(G2, sI)


def normalized_didv_with_error(V, current, current_error):
    """Feenstra-normalised conductance ``(dI/dV) / (I/V)`` and its one-sigma error.

    The normalisation divides by the current at the same point that the
    derivative is built from neighbouring points, so the numerator and
    denominator are correlated. The Jacobian below carries that correlation
    exactly instead of adding the two relative errors in quadrature.
    """
    V = np.asarray(V, dtype=float)
    I = np.asarray(current, dtype=float)
    sI = np.asarray(current_error, dtype=float)
    G = gradient_jacobian(V)
    dIdV = G @ I
    with np.errstate(divide='ignore', invalid='ignore'):
        norm = dIdV * V / I
        # d(norm_i)/d(I_j) = V_i G_ij / I_i  -  norm_i / I_i * delta_ij
        jac = (V / I)[:, None] * G
        jac[np.arange(len(V)), np.arange(len(V))] -= norm / I
        sigma = _propagate(jac, sI)
    return norm, sigma
