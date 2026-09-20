# ---
# description: |
#   Spot-check Gaussian + linear-baseline fits at a hand-picked list of
#   field values, with per-curve overrides on the fit parameters. The
#   workflow is: tune the per-H overrides interactively in the notebook,
#   eyeball the overlay plot, then save the resulting parameters to CSV.
#   This replaces the fully automated `extract_peak_features` sweep for
#   curves whose baseline / window has to be chosen by hand.
# entry_point: from scripts.peak_spotcheck import spot_check_fits
# dependencies:
#   - numpy, pandas, matplotlib, scripts.peak_features
# input: |
#   spot_check_fits(df, targets, defaults, axis_label, save_dir, save_stem,
#   V_lim=None, color_cycle=None). `targets` is a list of dicts; each entry
#   must have 'H' (target field) and may override any of 'fit_region',
#   'peak_bounds', 'fwhm_bounds', 'fine_half_width'. `defaults` supplies
#   the values for any key not overridden.
# process: |
#   For each target dict: merge defaults with per-curve overrides, pick
#   the row in df whose H is closest to target['H'], run fit_peak_features,
#   overlay (data, Gaussian+linear fit, fitted linear baseline, peak line)
#   on a single axis. Save the PNG to {save_dir}/{save_stem}.png and the
#   fitted parameters to {save_dir}/{save_stem}.csv.
# output: |
#   Returns a pandas dataframe with one row per target: H_target, H,
#   peak_pos_V, peak_pos_err, fwhm_V, fwhm_err, height, height_err,
#   rough_peak_V, ok, and the per-curve fit-config columns (fit_region_lo,
#   ..., fine_half_width). Side effects: matplotlib figure, PNG, CSV.
# last_updated: 2026-05-25
# ---
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.barrier_canting_fit import gauss_lin, normalised_didv
from scripts.peak_features import fit_peak_features


_FIT_KEYS = ('fit_region', 'peak_bounds', 'fwhm_bounds', 'fine_half_width')


def _resolve(target, defaults):
    """Return a (H, kwargs) tuple, with `kwargs` ready for fit_peak_features."""
    if not isinstance(target, dict) or 'H' not in target:
        raise ValueError("each target must be a dict with at least an 'H' key")
    kwargs = {k: defaults[k] for k in _FIT_KEYS if k in defaults}
    for k in _FIT_KEYS:
        if k in target:
            kwargs[k] = target[k]
    missing = [k for k in ('fit_region', 'peak_bounds', 'fwhm_bounds') if k not in kwargs]
    if missing:
        raise ValueError(f"missing fit parameters {missing} for target H={target['H']}")
    return float(target['H']), kwargs


def spot_check_fits(df, targets, defaults, axis_label, save_dir, save_stem,
                    V_lim=None, color_cycle=None,
                    voltage_col='voltage_smooth', current_col='current_smooth'):
    """Fit a selected list of curves, overlay them, and persist the parameters.

    Parameters
    ----------
    df : DataFrame
        IV dataframe with rows containing voltage_col, current_col, 'H'.
    targets : list of dict
        Each entry: {'H': H_target, optional overrides on 'fit_region',
        'peak_bounds', 'fwhm_bounds', 'fine_half_width'}.
    defaults : dict
        Default fit parameters used for any key not present in a target.
    axis_label : str
        e.g. 'H_Z' or 'H_y'; used in the legend labels.
    save_dir : Path-like
        Directory for the output PNG and CSV.
    save_stem : str
        Filename stem (no extension) for both files.
    V_lim : (V_min, V_max), optional
        X-axis limit for the overlay plot.
    color_cycle : sequence of colors, optional
        Defaults to scripts.utils.OKABE_ITO_CYCLE if available.

    Returns the fitted-parameters dataframe (also written to CSV).
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if color_cycle is None:
        try:
            from scripts.utils import OKABE_ITO_CYCLE
            color_cycle = OKABE_ITO_CYCLE
        except Exception:
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=200)
    rows = []

    for i, target in enumerate(targets):
        H_target, kw = _resolve(target, defaults)
        idx = (df['H'] - H_target).abs().idxmin()
        row = df.loc[idx]
        V, norm = normalised_didv(row[voltage_col], row[current_col])
        g = fit_peak_features(V, norm, **kw)

        color = color_cycle[i % len(color_cycle)]
        ax.plot(V, norm, 'o', color=color, markersize=2.5, alpha=0.45,
                label=fr'${axis_label}={row["H"]:+.2f}$ T')
        if g['popt'] is not None and g['window'] is not None:
            Vmin, Vmax = g['window']
            Vfit = np.linspace(Vmin, Vmax, 400)
            ax.plot(Vfit, gauss_lin(Vfit, *g['popt']), '-', color=color, lw=1.4)
            A, V0, sigma, m_lin, c_lin = g['popt']
            ax.plot(Vfit, m_lin * Vfit + c_lin, ':', color=color, lw=0.9, alpha=0.7)
        if g['ok'] and np.isfinite(g['peak_pos_V']):
            ax.axvline(g['peak_pos_V'], color=color, ls='--', lw=0.9, alpha=0.85)

        rows.append({
            'H_target':     H_target,
            'H':            float(row['H']),
            'peak_pos_V':   g['peak_pos_V'],
            'peak_pos_err': g['peak_pos_err'],
            'fwhm_V':       g['fwhm_V'],
            'fwhm_err':     g['fwhm_err'],
            'height':       g['height'],
            'height_err':   g['height_err'],
            'rough_peak_V': g['rough_peak_V'],
            'ok':           g['ok'],
            'fit_region_lo':   kw['fit_region'][0],
            'fit_region_hi':   kw['fit_region'][1],
            'peak_bounds_lo':  kw['peak_bounds'][0],
            'peak_bounds_hi':  kw['peak_bounds'][1],
            'fwhm_bounds_lo':  kw['fwhm_bounds'][0],
            'fwhm_bounds_hi':  kw['fwhm_bounds'][1],
            'fine_half_width': kw.get('fine_half_width', np.nan),
        })

    ax.set_xlabel(r'$V_{\mathrm{bias}}$ (V)')
    ax.set_ylabel(r'$(dI/dV)/(I/V)$')
    if V_lim is not None:
        ax.set_xlim(*V_lim)
    ax.legend(loc='upper left', fontsize=8, frameon=True)
    fig.tight_layout()
    png_path = save_dir / f'{save_stem}.png'
    csv_path = save_dir / f'{save_stem}.csv'
    fig.savefig(png_path, dpi=200)
    plt.show()

    out = pd.DataFrame(rows)
    out.to_csv(csv_path, index=False)
    print(f'spot-check: {int(out["ok"].sum())}/{len(out)} fits ok  ->  {png_path.name}, {csv_path.name}')
    return out


__all__ = ['spot_check_fits']
