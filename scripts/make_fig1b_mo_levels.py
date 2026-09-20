"""Candidate panel (b): bonding/antibonding molecular-orbital level diagram.

Draws the band-edge hybridization mechanism as a compact two-level molecular-
orbital diagram, the form in which the v4 interpretation states it (the two
degenerate layer band-edge states hybridize through t_perp(theta) = t_0 cos(theta/2);
the bonding state, the band edge, shifts by -|t_perp|). Left (AFM): the two layer
band-edge states are antiparallel-spin, the coupling t_perp = 0, the levels stay
degenerate and unsplit, and the band edge sits high (Phi_AFM). Right (FM):
parallel spins switch the coupling on, the levels split into bonding and
antibonding, and the bonding state is the lowered band edge (Phi_FM). All
energies are symbolic; no absolute meV is shown, per the v4 interpretation.

Input:
    No data files. Optional CLI flags --out and --dpi. Colours/style come from
    scripts.utils.notebook_setup (Okabe-Ito) when importable, else a fallback.

Process:
    Pure matplotlib: layer-moment spin arrows, level segments, hybridization
    split arrows, Phi brackets to the Fermi level, and the V matrix-element label.

Output:
    A single PNG to --out (default output/fig1b_candidate_mo_levels.png).
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


EF, E0, TPERP = 0.0, 1.0, 0.35
BOND, ANTI = E0 - TPERP, E0 + TPERP


def draw_cell(ax, state, c_up, c_dn, show_label=True):
    ax.set_xlim(-0.12, 1.06)
    ax.set_ylim(-0.42, 1.95)
    ax.axis('off')
    ax.plot([0.04, 0.96], [EF, EF], ls=(0, (5, 3)), color='k', lw=1.3)
    ax.text(0.97, EF, r'$E_\mathrm{F}$', va='center', ha='left')

    if show_label:
        label = 'AFM (antiparallel)' if state == 'AFM' else 'FM (parallel)'
        ax.text(0.5, 1.88, label, ha='center', va='top')

    if state == 'AFM':
        ax.annotate('', xy=(0.0, 1.62), xytext=(0.0, -0.28),
                    arrowprops=dict(arrowstyle='-|>', color='k', lw=1.4))
        ax.text(-0.08, 0.68, r'Energy, $E$', rotation=90, va='center', ha='center')
        ax.plot([0.20, 0.44], [E0, E0], color=c_up, lw=3.4, solid_capstyle='round')
        ax.plot([0.56, 0.80], [E0, E0], color=c_dn, lw=3.4, solid_capstyle='round')
        ax.annotate('', xy=(0.12, E0), xytext=(0.12, EF),
                    arrowprops=dict(arrowstyle='<->', color='k', lw=1.5))
        ax.text(0.155, E0 / 2, r'$\Phi_\mathrm{AFM}$', va='center', ha='left')
        ax.text(0.5, E0 + 0.14, r'degenerate, $t_\perp=0$', ha='center', va='bottom')
    else:
        ax.plot([0.18, 0.42], [E0, E0], color=c_up, lw=1.6, ls=(0, (3, 2)), alpha=0.55)
        ax.plot([0.58, 0.82], [E0, E0], color=c_up, lw=1.6, ls=(0, (3, 2)), alpha=0.55)
        ax.plot([0.32, 0.68], [BOND, BOND], color=c_up, lw=3.6, solid_capstyle='round')
        ax.plot([0.32, 0.68], [ANTI, ANTI], color=c_up, lw=2.4, solid_capstyle='round', alpha=0.85)
        ax.annotate('', xy=(0.50, BOND + 0.03), xytext=(0.50, E0 - 0.03),
                    arrowprops=dict(arrowstyle='-|>', color='0.4', lw=1.2))
        ax.annotate('', xy=(0.50, ANTI - 0.03), xytext=(0.50, E0 + 0.03),
                    arrowprops=dict(arrowstyle='-|>', color='0.4', lw=1.2))
        ax.text(0.70, ANTI, 'antibonding', va='center', ha='left')
        ax.text(0.70, BOND, 'bonding (edge)', va='center', ha='left')
        ax.annotate('', xy=(0.12, BOND), xytext=(0.12, EF),
                    arrowprops=dict(arrowstyle='<->', color='k', lw=1.5))
        ax.text(0.155, BOND / 2, r'$\Phi_\mathrm{FM}$', va='center', ha='left')
        ax.text(0.5, ANTI + 0.14, r'$t_\perp(\theta)=t_0\cos(\theta/2)=t_0\,m$', ha='center', va='bottom')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    p.add_argument('--out', type=Path,
                   default=Path(__file__).resolve().parent.parent / 'output' / 'fig1b_candidate_mo_levels.png')
    p.add_argument('--dpi', type=int, default=300)
    args = p.parse_args()

    oi = load_style()
    c_up, c_dn = oi['vermillion'], oi['blue']

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.5, 4.4))
    draw_cell(axL, 'AFM', c_up, c_dn)
    draw_cell(axR, 'FM', c_up, c_dn)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    print(f'[OK] saved {args.out}')


if __name__ == '__main__':
    main()
