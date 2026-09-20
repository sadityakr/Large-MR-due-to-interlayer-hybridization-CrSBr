"""Render the redesigned panel (b) of Figure 1: the band-edge spin-filter schematic.

Draws a three-cell, publication-quality energy-band schematic for the CrSBr
vertical tunnel junction, replacing the previous hand-drawn panel (b). Cell (i)
shows the field-polarised ferromagnetic (FM) state as a uniform low barrier with
the exchange-split spin-up/spin-down conduction-band edges; cell (ii) shows the
A-type antiferromagnetic (AFM) ground state as a layer-alternating (crenellated)
higher effective barrier; cell (iii) shows the predicted M-shaped TMR(V) with
peaks at eV* ~ Phi_FM. All barrier heights are symbolic (Phi_FM, Phi_AFM,
Delta_ex) with no absolute meV values, consistent with the v4 interpretation
which retracts the absolute calibrated barrier.

Input:
    No data files. Optional CLI flags --out (output PNG path) and --dpi.
    Colours and matplotlib style are taken from scripts.utils.notebook_setup
    (Okabe-Ito palette) when importable, with a local fallback.

Process:
    Pure matplotlib drawing: electrode patches, conduction-band-edge lines
    (spin-up vermillion/solid, spin-down blue/dashed), bias-tilted bands,
    tunnelling arrows, layer-magnetisation arrows, and a schematic TMR(V) curve.

Output:
    A single PNG written to --out (default output/figure1b_band_diagram.png).
    Prints the saved path to stdout.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "matplotlib>=3.7",
#   "numpy>=1.24",
# ]
# ///

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Style: reuse the project's Okabe-Ito publication style when importable.
# --------------------------------------------------------------------------- #
def load_style() -> dict:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from scripts.utils.notebook_setup import configure_plot_style, OKABE_ITO
        configure_plot_style()
        return OKABE_ITO
    except Exception:
        return {
            'orange': '#E69F00', 'sky_blue': '#56B4E9', 'bluish_green': '#009E73',
            'yellow': '#F0E442', 'blue': '#0072B2', 'vermillion': '#D55E00',
            'reddish_purple': '#CC79A7', 'soft_violet': '#7B5EA7',
        }


# --------------------------------------------------------------------------- #
# Layout constants (schematic energy units; no absolute meV).
# --------------------------------------------------------------------------- #
X_LE = (0.00, 0.18)     # left electrode span
X_BAR = (0.18, 0.82)    # barrier span
X_RE = (0.82, 1.00)     # right electrode span
EF_L = 0.0              # left Fermi level
EV = 0.30              # applied bias (band tilt magnitude)
EF_R = EF_L - EV        # right Fermi level (under +V)
PHI_FM = 1.00           # FM (majority/low) barrier height
DELTA = 0.55           # conduction-band exchange splitting Delta_ex
PHI_HI = PHI_FM + DELTA  # minority edge / effective AFM barrier
Y_FILL = -1.10          # bottom of electrode fill
YLIM = (-1.55, 2.35)
N_LAYERS = 4
SEG_CENTERS = [X_BAR[0] + (k + 0.5) * (X_BAR[1] - X_BAR[0]) / N_LAYERS
               for k in range(N_LAYERS)]

ELEC_GRAY = '#d3d3d3'
EDGE_GRAY = '#9a9a9a'


def tilted_square_wave(x0, x1, levels, tilt):
    """Polyline for a bias-tilted square wave: piecewise levels minus a linear ramp."""
    n = len(levels)
    seg = (x1 - x0) / n
    xs, ys = [], []
    for k, lv in enumerate(levels):
        xa, xb = x0 + k * seg, x0 + (k + 1) * seg
        xs += [xa, xb]
        ys += [lv - tilt * (xa - x0) / (x1 - x0),
               lv - tilt * (xb - x0) / (x1 - x0)]
    return np.array(xs), np.array(ys)


def draw_electrodes(ax):
    """Two FLG electrodes filled to their (biased) Fermi levels, with E_F dashes."""
    for (x0, x1), ef in ((X_LE, EF_L), (X_RE, EF_R)):
        ax.fill_between([x0, x1], Y_FILL, ef, color=ELEC_GRAY,
                        edgecolor=EDGE_GRAY, linewidth=1.2, zorder=1)
        ax.plot([x0, x1], [ef, ef], color='black', lw=1.4, ls=(0, (5, 3)), zorder=4)
    ax.text(np.mean(X_LE), (Y_FILL + EF_L) / 2, 'FLG', ha='center', va='center', color='#444')
    ax.text(np.mean(X_RE), (Y_FILL + EF_R) / 2, 'FLG', ha='center', va='center', color='#444')


def draw_magnetisation(ax, pattern, c_up, c_dn):
    """Layer-moment arrows above the barrier (pattern: list of +1/-1)."""
    y0, h = 1.85, 0.42
    for xc, s in zip(SEG_CENTERS, pattern):
        if s > 0:
            ax.annotate('', xy=(xc, y0 + h), xytext=(xc, y0),
                        arrowprops=dict(arrowstyle='-|>', color=c_up, lw=2.4))
        else:
            ax.annotate('', xy=(xc, y0), xytext=(xc, y0 + h),
                        arrowprops=dict(arrowstyle='-|>', color=c_dn, lw=2.4))


def cell_fm(ax, c_up, c_dn):
    draw_electrodes(ax)
    # exchange-split conduction-band edges (bias-tilted, uniform)
    ax.plot(list(X_BAR), [PHI_FM, PHI_FM - EV], color=c_up, lw=3.0, solid_capstyle='round', zorder=5)
    ax.plot(list(X_BAR), [PHI_HI, PHI_HI - EV], color=c_dn, lw=2.4, ls=(0, (4, 2)), zorder=5)
    # Delta_ex between the two spin edges
    xd = X_BAR[0] + 0.13
    ax.annotate('', xy=(xd, PHI_HI - EV * (xd - X_BAR[0]) / (X_BAR[1] - X_BAR[0])),
                xytext=(xd, PHI_FM - EV * (xd - X_BAR[0]) / (X_BAR[1] - X_BAR[0])),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.4))
    ax.text(xd + 0.015, (PHI_FM + PHI_HI) / 2 - 0.07, r'$\Delta_\mathrm{ex}$', va='center', ha='left')
    # Phi_FM bracket at the left interface
    xb = X_BAR[0]
    ax.annotate('', xy=(xb, PHI_FM), xytext=(xb, EF_L),
                arrowprops=dict(arrowstyle='<->', color=c_up, lw=1.6))
    ax.text(xb - 0.015, PHI_FM / 2, r'$\Phi_\mathrm{FM}$', va='center', ha='right', color=c_up)
    # layer moments (all up) + efficient tunnelling
    draw_magnetisation(ax, [1, 1, 1, 1], c_up, c_dn)
    ax.annotate('', xy=(X_BAR[1], EF_L), xytext=(X_BAR[0], EF_L),
                arrowprops=dict(arrowstyle='-|>', color=c_up, lw=3.2))


def cell_afm(ax, c_up, c_dn):
    draw_electrodes(ax)
    up_levels = [PHI_FM, PHI_HI, PHI_FM, PHI_HI]
    dn_levels = [PHI_HI, PHI_FM, PHI_HI, PHI_FM]
    xu, yu = tilted_square_wave(*X_BAR, up_levels, EV)
    xd, yd = tilted_square_wave(*X_BAR, dn_levels, EV)
    ax.plot(xd, yd, color=c_dn, lw=2.0, ls=(0, (4, 2)), zorder=4, alpha=0.7)
    ax.plot(xu, yu, color=c_up, lw=3.0, solid_capstyle='round', zorder=5)
    # effective AFM barrier reference line
    ax.plot(list(X_BAR), [PHI_HI, PHI_HI], color='black', lw=1.2, ls=':', zorder=3)
    ax.annotate('', xy=(X_BAR[0], PHI_HI), xytext=(X_BAR[0], EF_L),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.6))
    ax.text(X_BAR[0] - 0.015, PHI_HI / 2, r'$\Phi_\mathrm{AFM}$', va='center', ha='right')
    # alternating moments + suppressed tunnelling
    draw_magnetisation(ax, [1, -1, 1, -1], c_up, c_dn)
    ax.annotate('', xy=(0.52, EF_L), xytext=(X_BAR[0], EF_L),
                arrowprops=dict(arrowstyle='-|>', color=c_up, lw=1.6, alpha=0.5))
    ax.text(0.60, EF_L + 0.02, 'blocked', ha='left', va='center', color='#777')


def cell_tmr(ax, c_curve):
    vmax, vstar, sigma, peak = 1.0, 0.50, 0.25, 100.0
    v = np.linspace(-vmax, vmax, 600)
    tmr = peak * (np.exp(-((v - vstar) ** 2) / (2 * sigma ** 2))
                  + np.exp(-((v + vstar) ** 2) / (2 * sigma ** 2)))
    ax.plot(v, tmr, color=c_curve, lw=2.6)
    for s in (-vstar, vstar):
        ax.axvline(s, color='black', lw=1.0, ls=':')
    ax.text(vstar, peak * 1.06, r'$eV^{*}\!\approx\!\Phi_\mathrm{FM}$', ha='center', va='bottom')
    ax.set_xlim(-vmax, vmax)
    ax.set_ylim(0, peak * 1.18)
    ax.set_xticks([-vstar, 0, vstar])
    ax.set_xticklabels([r'$-V^{*}$', '0', r'$V^{*}$'])
    ax.set_yticks([])
    ax.set_xlabel(r'Bias, $V$')
    ax.set_ylabel('TMR (%)')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    p.add_argument('--out', type=Path,
                   default=Path(__file__).resolve().parent.parent / 'output' / 'figure1b_band_diagram.png',
                   help='Output PNG path.')
    p.add_argument('--dpi', type=int, default=300, help='Output resolution (dpi).')
    args = p.parse_args()

    oi = load_style()
    c_up, c_dn = oi['vermillion'], oi['blue']
    c_curve = oi['bluish_green']

    fig = plt.figure(figsize=(11.0, 3.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.92], wspace=0.28)
    ax_fm, ax_afm, ax_tmr = (fig.add_subplot(gs[0, i]) for i in range(3))

    for ax in (ax_fm, ax_afm):
        ax.set_xlim(-0.06, 1.0)
        ax.set_ylim(*YLIM)
        ax.axis('off')

    cell_fm(ax_fm, c_up, c_dn)
    cell_afm(ax_afm, c_up, c_dn)
    cell_tmr(ax_tmr, c_curve)

    # energy axis arrow on the first cell
    ax_fm.annotate('', xy=(-0.04, 1.7), xytext=(-0.04, EF_L - 0.2),
                   arrowprops=dict(arrowstyle='-|>', color='black', lw=1.6))
    ax_fm.text(-0.075, 0.85, r'Energy, $E$', rotation=90, va='center', ha='center')

    # state labels
    ax_fm.text(0.5, -1.45, 'FM (parallel)', ha='center', va='center')
    ax_afm.text(0.5, -1.45, 'AFM (ground state)', ha='center', va='center')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    print(f'[OK] saved {args.out}')


if __name__ == '__main__':
    main()
