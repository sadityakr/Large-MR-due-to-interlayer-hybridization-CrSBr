"""Candidate panel (b): side-by-side E-vs-k band structure, AFM versus FM.

Draws schematic conduction/valence bands for the CrSBr barrier in the AFM ground
state and the field-polarised FM state, modelled on the dominant published
convention (Wilson et al., Nat. Mater. 2021, Fig. 3; arXiv:2502.03739, Fig. 7).
In AFM the band-edge bands are spin-degenerate and the conduction-band minimum
(CBM) sits high, setting a large barrier Phi_AFM. In FM the spin-allowed
interlayer hybridization splits the bands (spin-up solid, spin-down dashed) and
pushes the CBM down, lowering the barrier to Phi_FM. The band-edge shift is drawn
symbolically; no absolute meV is shown, per the v4 interpretation.

Input:
    No data files. Optional CLI flags --out and --dpi. Colours/style come from
    scripts.utils.notebook_setup (Okabe-Ito) when importable, else a fallback.

Process:
    Pure matplotlib: schematic parabolic conduction and valence bands, a dashed
    Fermi level, Phi brackets at the band edge, a band-edge-shift arrow, and a
    spin-up/spin-down legend.

Output:
    A single PNG to --out (default output/fig1b_candidate_band_structure.png).
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
from matplotlib.lines import Line2D


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


KC, KV = 1.1, 0.9  # conduction / valence band curvatures


def draw_band(ax, state, c_up, c_dn):
    k = np.linspace(0, 1, 200)
    ax.set_xlim(0, 1)
    ax.set_ylim(-2.0, 2.0)
    ax.plot([0, 1], [0, 0], ls=(0, (5, 3)), color='k', lw=1.2, zorder=2)
    ax.text(0.98, 0.07, r'$E_\mathrm{F}$', va='bottom', ha='right')
    ax.set_xticks([0, 1])
    ax.set_xticklabels([r'$\Gamma$', r'$X$'])
    ax.set_yticks([])
    ax.set_xlabel(r'Wavevector, $k$')
    label = 'AFM (ground state)' if state == 'AFM' else 'FM (field-polarised)'
    ax.text(0.5, 1.78, label, ha='center', va='top')

    if state == 'AFM':
        cbm, vbm = 1.0, -1.4
        ax.plot(k, cbm + KC * k ** 2, color='0.25', lw=2.8, zorder=4)
        ax.plot(k, vbm - KV * k ** 2, color='0.25', lw=2.8, zorder=4)
        ax.annotate('', xy=(0.07, cbm), xytext=(0.07, 0),
                    arrowprops=dict(arrowstyle='<->', color='k', lw=1.5))
        ax.text(0.10, cbm / 2, r'$\Phi_\mathrm{AFM}$', va='center', ha='left')
        ax.text(0.62, 0.45, 'spin-degenerate', ha='center', va='center')
    else:
        up_cbm, dn_cbm = 0.6, 1.05
        up_vbm, dn_vbm = -1.30, -1.55
        # faint AFM reference + band-edge-shift arrow
        ax.plot([0.0, 0.34], [1.0, 1.0], color='0.55', lw=1.1, ls=':', zorder=3)
        ax.annotate('', xy=(0.20, up_cbm), xytext=(0.20, 1.0),
                    arrowprops=dict(arrowstyle='-|>', color='k', lw=1.4))
        ax.text(0.24, 0.82, 'band-edge\nshift', va='center', ha='left')
        ax.plot(k, up_cbm + KC * k ** 2, color=c_up, lw=2.9, zorder=5)
        ax.plot(k, dn_cbm + KC * k ** 2, color=c_dn, lw=2.5, ls=(0, (4, 2)), zorder=5)
        ax.plot(k, up_vbm - KV * k ** 2, color=c_up, lw=2.5, zorder=4)
        ax.plot(k, dn_vbm - KV * k ** 2, color=c_dn, lw=2.1, ls=(0, (4, 2)), zorder=4)
        ax.annotate('', xy=(0.07, up_cbm), xytext=(0.07, 0),
                    arrowprops=dict(arrowstyle='<->', color=c_up, lw=1.5))
        ax.text(0.10, up_cbm / 2, r'$\Phi_\mathrm{FM}$', va='center', ha='left', color=c_up)
        handles = [Line2D([0], [0], color=c_up, lw=2.9, label='spin-up'),
                   Line2D([0], [0], color=c_dn, lw=2.5, ls=(0, (4, 2)), label='spin-down')]
        ax.legend(handles=handles, loc='lower right')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    p.add_argument('--out', type=Path,
                   default=Path(__file__).resolve().parent.parent / 'output' / 'fig1b_candidate_band_structure.png')
    p.add_argument('--dpi', type=int, default=300)
    args = p.parse_args()

    oi = load_style()
    c_up, c_dn = oi['vermillion'], oi['blue']

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.0, 4.4), sharey=True)
    draw_band(axL, 'AFM', c_up, c_dn)
    draw_band(axR, 'FM', c_up, c_dn)
    axL.set_ylabel(r'Energy, $E$ (arb. u.)')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    print(f'[OK] saved {args.out}')


if __name__ == '__main__':
    main()
