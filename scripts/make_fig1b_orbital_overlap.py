"""Candidate panel (b): real-space orbital overlap + bonding/antibonding levels.

Draws the band-edge hybridization mechanism for the CrSBr vertical junction as a
2x2 schematic. Top row: two CrSBr layers with their band-edge (Cr e_g) orbital
lobes facing the van der Waals gap and a bridging Br; in the field-polarised FM
state the same-spin lobes hybridize across the gap (overlap), whereas in the AFM
ground state the opposite-spin lobes cannot (spin-blocked). Bottom row: the
resulting energy levels, degenerate and unsplit in AFM (high band edge, Phi_AFM)
versus bonding/antibonding split in FM, the bonding state being the lowered band
edge (Phi_FM = Phi_AFM - |V|). All energies are symbolic (no absolute meV), per
the v4 interpretation that retracts the absolute calibrated barrier.

Input:
    No data files. Optional CLI flags --out and --dpi. Colours/style come from
    scripts.utils.notebook_setup (Okabe-Ito) when importable, else a fallback.

Process:
    Pure matplotlib drawing with patches (layer slabs, orbital-lobe ellipses,
    Br atom), spin arrows, overlap/blocked markers, and a bonding/antibonding
    level diagram with Phi brackets.

Output:
    A single PNG to --out (default output/fig1b_candidate_orbital_overlap.png).
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

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle


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


EF, E0, V = 0.0, 1.0, 0.35
BOND, ANTI = E0 - V, E0 + V


def _spin_arrow(ax, xc, yc, up, color, h=0.05):
    if up:
        ax.annotate('', xy=(xc, yc + h), xytext=(xc, yc - h),
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=2.4))
    else:
        ax.annotate('', xy=(xc, yc - h), xytext=(xc, yc + h),
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=2.4))


def draw_realspace(ax, state, c_up, c_dn, c_br):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    label = 'AFM (ground state)' if state == 'AFM' else 'FM (field-polarised)'
    ax.text(0.5, 0.96, label, ha='center', va='top')

    for yb in (0.60, 0.24):
        ax.add_patch(Rectangle((0.12, yb), 0.76, 0.16, facecolor='#e9e9e9',
                               edgecolor='#9a9a9a', lw=1.2, zorder=1))

    top_up = True
    bot_up = (state == 'FM')
    top_col = c_up
    bot_col = c_up if bot_up else c_dn

    _spin_arrow(ax, 0.20, 0.68, top_up, top_col)
    _spin_arrow(ax, 0.20, 0.32, bot_up, bot_col)

    a = 0.65 if state == 'FM' else 0.50
    # lobes face the gap; pulled slightly apart in AFM (weaker overlap)
    yt, yb = (0.55, 0.45) if state == 'FM' else (0.57, 0.43)
    ax.add_patch(Ellipse((0.45, yt), 0.13, 0.17, facecolor=top_col, alpha=a,
                         edgecolor=top_col, lw=1.5, zorder=3))
    ax.add_patch(Ellipse((0.45, yb), 0.13, 0.17, facecolor=bot_col, alpha=a,
                         edgecolor=bot_col, lw=1.5, zorder=3))
    ax.add_patch(Ellipse((0.45, 0.50), 0.045, 0.05, facecolor=c_br,
                         edgecolor='k', lw=0.8, zorder=5))
    ax.text(0.45 + 0.05, 0.50, 'Br', va='center', ha='left')

    if state == 'FM':
        ax.add_patch(Ellipse((0.45, 0.50), 0.16, 0.13, facecolor=c_up, alpha=0.22,
                             edgecolor='none', zorder=2))
        ax.annotate('', xy=(0.70, 0.59), xytext=(0.70, 0.41),
                    arrowprops=dict(arrowstyle='<->', color='k', lw=1.4))
        ax.text(0.72, 0.50, r'$t_\perp$', va='center', ha='left')
        ax.text(0.45, 0.18, 'spin-allowed overlap', ha='center', va='center')
    else:
        ax.plot([0.36, 0.54], [0.50, 0.50], color='0.45', lw=1.4, ls=(0, (2, 2)))
        ax.text(0.45, 0.18, 'spin-blocked', ha='center', va='center')


def draw_levels(ax, state, c_up, c_dn):
    ax.set_xlim(-0.12, 1.06)
    ax.set_ylim(-0.40, 1.78)
    ax.axis('off')
    ax.plot([0.04, 0.96], [EF, EF], ls=(0, (5, 3)), color='k', lw=1.3)
    ax.text(0.97, EF, r'$E_\mathrm{F}$', va='center', ha='left')

    if state == 'AFM':
        ax.annotate('', xy=(0.0, 1.55), xytext=(0.0, -0.25),
                    arrowprops=dict(arrowstyle='-|>', color='k', lw=1.4))
        ax.text(-0.08, 0.65, r'Energy, $E$', rotation=90, va='center', ha='center')
        ax.plot([0.20, 0.44], [E0, E0], color=c_up, lw=3.2, solid_capstyle='round')
        ax.plot([0.56, 0.80], [E0, E0], color=c_dn, lw=3.2, solid_capstyle='round')
        ax.annotate('', xy=(0.12, E0), xytext=(0.12, EF),
                    arrowprops=dict(arrowstyle='<->', color='k', lw=1.5))
        ax.text(0.155, E0 / 2, r'$\Phi_\mathrm{AFM}$', va='center', ha='left')
        ax.text(0.5, E0 + 0.13, r'degenerate, $V=0$', ha='center', va='bottom')
    else:
        ax.plot([0.18, 0.42], [E0, E0], color=c_up, lw=1.5, ls=(0, (3, 2)), alpha=0.55)
        ax.plot([0.58, 0.82], [E0, E0], color=c_up, lw=1.5, ls=(0, (3, 2)), alpha=0.55)
        ax.plot([0.32, 0.68], [BOND, BOND], color=c_up, lw=3.4, solid_capstyle='round')
        ax.plot([0.32, 0.68], [ANTI, ANTI], color=c_up, lw=2.2, solid_capstyle='round', alpha=0.85)
        ax.annotate('', xy=(0.50, BOND + 0.03), xytext=(0.50, E0 - 0.03),
                    arrowprops=dict(arrowstyle='-|>', color='0.4', lw=1.2))
        ax.annotate('', xy=(0.50, ANTI - 0.03), xytext=(0.50, E0 + 0.03),
                    arrowprops=dict(arrowstyle='-|>', color='0.4', lw=1.2))
        ax.text(0.70, ANTI, 'antibonding', va='center', ha='left')
        ax.text(0.70, BOND, 'bonding (edge)', va='center', ha='left')
        ax.annotate('', xy=(0.12, BOND), xytext=(0.12, EF),
                    arrowprops=dict(arrowstyle='<->', color='k', lw=1.5))
        ax.text(0.155, BOND / 2, r'$\Phi_\mathrm{FM}$', va='center', ha='left')
        ax.text(0.5, ANTI + 0.13, r'$V=t_\perp\cos(\theta/2)=t_\perp m$', ha='center', va='bottom')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    p.add_argument('--out', type=Path,
                   default=Path(__file__).resolve().parent.parent / 'output' / 'fig1b_candidate_orbital_overlap.png')
    p.add_argument('--dpi', type=int, default=300)
    args = p.parse_args()

    oi = load_style()
    c_up, c_dn, c_br = oi['vermillion'], oi['blue'], oi['reddish_purple']

    fig = plt.figure(figsize=(8.5, 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.02], hspace=0.10, wspace=0.16)
    draw_realspace(fig.add_subplot(gs[0, 0]), 'AFM', c_up, c_dn, c_br)
    draw_realspace(fig.add_subplot(gs[0, 1]), 'FM', c_up, c_dn, c_br)
    draw_levels(fig.add_subplot(gs[1, 0]), 'AFM', c_up, c_dn)
    draw_levels(fig.add_subplot(gs[1, 1]), 'FM', c_up, c_dn)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    print(f'[OK] saved {args.out}')


if __name__ == '__main__':
    main()
