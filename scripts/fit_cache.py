# ---
# description: |
#   Disk-cache wrapper for the global shared-sigma (shared-FWHM) fit used
#   by the barrier_vs_canting_V2 notebooks. The fit itself takes 3-4 min
#   per notebook; this wrapper pickles the (curves, fit) tuple so a kernel
#   restart can reload in <1 s. Invalidation is manual: delete the cache
#   file (or pass force=True) to recompute. This was an explicit design
#   choice -- mtime / config-hash invalidation was rejected as too eager
#   when re-running notebooks just to tweak plot styling.
# entry_point: from scripts.fit_cache import load_or_compute_shared_fit
# dependencies:
#   - numpy, pandas (transitively, via scripts.global_shared_fwhm)
#   - scripts.global_shared_fwhm
# input: |
#   df (IV_gaussian dataframe), axis ('c' or 'b'), temperature (K),
#   cache_dir (Path), tight_half_width, sigma_init. Optional force=True
#   bypasses the cache.
# process: |
#   Builds cache_path = {cache_dir}/{axis}_{T}K_fit.pkl. If it exists
#   and force is False, unpickles and returns (curves, fit). Otherwise
#   runs prepare_curves + fit_shared_sigma_global, pickles the result,
#   and returns it. Prints a one-line status either way.
# output: |
#   (curves, fit) tuple identical to running prepare_curves and
#   fit_shared_sigma_global directly. Side effect: pickle file at
#   cache_path on a cache miss.
# last_updated: 2026-05-27
# ---
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from scripts.global_shared_fwhm import prepare_curves, fit_shared_sigma_global

FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))


def load_or_compute_shared_fit(df, axis, temperature, cache_dir,
                                tight_half_width, sigma_init, force=False):
    """Load cached (curves, fit) if available, else compute and cache.

    Invalidation is manual: delete the cache file or pass force=True.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f'{axis}_{int(temperature)}K_fit.pkl'

    if cache_path.exists() and not force:
        with open(cache_path, 'rb') as f:
            payload = pickle.load(f)
        curves = payload['curves']
        fit = payload['fit']
        sigma_mV = 1000 * fit['sigma']
        fwhm_mV = 1000 * FWHM_FACTOR * fit['sigma']
        print(f'[cache hit]  {cache_path.name}: sigma = {sigma_mV:.2f} mV  '
              f'-> FWHM = {fwhm_mV:.2f} mV  ({len(curves)} curves)')
        return curves, fit

    print(f'[cache miss] computing shared-sigma fit for {axis}-axis at {temperature} K ...')
    curves = prepare_curves(df, tight_half_width=tight_half_width)
    fit = fit_shared_sigma_global(curves, sigma_init=sigma_init)

    with open(cache_path, 'wb') as f:
        pickle.dump({'curves': curves, 'fit': fit}, f)
    sigma_mV = 1000 * fit['sigma']
    fwhm_mV = 1000 * FWHM_FACTOR * fit['sigma']
    print(f'[cached]     {cache_path.name}: sigma = {sigma_mV:.2f} mV  '
          f'-> FWHM = {fwhm_mV:.2f} mV  ({len(curves)} curves)')
    return curves, fit


__all__ = ['load_or_compute_shared_fit']
