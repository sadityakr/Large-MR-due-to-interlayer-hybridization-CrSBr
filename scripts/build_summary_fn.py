"""Merge the FN summary into Summary_T_dependence.ipynb.

Appends, before the final save cell: V_T^AFM(T) and calibration-factor c(T)
trends (calibrated 20-80 K window), an explicitly separated uncalibrated
high-T block (T > 80 K, where the AFM FN minimum is lost, reported as
relative peak splitting only), a field-resolved Fowler-Nordheim grid for
every temperature, and an AFM/FM FN endpoint overview vs T. Reads the new
calibration columns (c_calib_*, V_T_afm_*, calib_ok_*) the per-T notebooks
now write, and the c-axis IV_gaussian dataframes for the FN plots.

Input:
    Summary_T_dependence.ipynb and the per-T fit_summary CSVs / IV_gaussian
    dataframes already present under output/IV_H_scans/.

Process:
    Locate the "## 6. Save combined CSV" cell and insert the FN-summary
    cells just before it via nbformat; refresh the intro. Idempotent: it
    removes any previously inserted FN-summary cells (tagged) before adding.

Output:
    Overwrites Summary_T_dependence.ipynb in place (code input collapsed by
    default, matching the per-T notebooks). Prints the path.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = ["nbformat>=5.10"]
# ///
from __future__ import annotations

from pathlib import Path

import nbformat

NB = (Path(__file__).resolve().parents[1] / "notebooks" / "Device_2"
      / "barrier and canting analysis" / "Summary_T_dependence.ipynb")

TAG = "fn-summary-merge"   # marks inserted cells for idempotent rebuilds


def md(text):
    c = nbformat.v4.new_markdown_cell(text)
    c.metadata["tags"] = [TAG]
    return c


def code(text):
    c = nbformat.v4.new_code_cell(text)
    c.metadata["tags"] = [TAG]
    c.metadata.setdefault("jupyter", {})["source_hidden"] = True
    return c


INTRO = (
    "# Temperature summary (V5): calibrated barrier heights + FN summary\n\n"
    "Aggregates the per-temperature `fit_summary_{T}K.csv` files written by the "
    "`Spectroscopy and FN analysis_{T}K.ipynb` notebooks. Each per-T CSV holds the "
    "Method-2 binned $\\Phi_\\mathrm{AFM}$, $\\Phi_\\mathrm{FM}$, $\\Delta_\\mathrm{ex}$ "
    "for both axes, the below-saturation $\\cos(\\theta/2)$ and $\\sin^2(\\theta/2)$ fit "
    "parameters, and the AFM-anchored transition-voltage calibration "
    "($c = V_\\mathrm{peak}^\\mathrm{AFM}/V_T^\\mathrm{AFM}$, $V_T^\\mathrm{AFM}$, "
    "`calib_ok`).\n\n"
    "**Calibrated window:** the absolute $\\Phi(T)$ plots use only $T \\leq 80$ K, where "
    "a clean AFM FN minimum resolves. At 90 and 100 K the field-emission regime is lost, "
    "so those points carry no absolute calibration and are reported as relative peak "
    "splitting only (Section 7c)."
)

SEC7_MD = (
    "## 7. FN transition voltage and calibration factor vs $T$\n\n"
    "### 7a. $eV_T^\\mathrm{AFM} = \\Phi_\\mathrm{AFM}(T)$\n\n"
    "The AFM transition voltage measured from the FN minimum, which anchors the "
    "calibration ($\\Phi_\\mathrm{AFM} = eV_T^\\mathrm{AFM}$ by construction). Only "
    "temperatures with a resolved minimum appear."
)

SEC7A_CODE = r'''Tv = summary['temperature_K'].values
m_c = np.isfinite(summary['V_T_afm_c_meV'].values)
m_b = np.isfinite(summary['V_T_afm_b_meV'].values)

fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
ax.plot(Tv[m_c], summary['V_T_afm_c_meV'].values[m_c], 'o-',
        color=OKABE_ITO_CYCLE[1], label='c-axis')
ax.plot(Tv[m_b], summary['V_T_afm_b_meV'].values[m_b], 's-',
        color=OKABE_ITO_CYCLE[5], mfc='white', label='b-axis')
ax.set_xlabel(r'$T$ (K)')
ax.set_ylabel(r'$eV_T^\mathrm{AFM} = \Phi_\mathrm{AFM}$ (meV)')
ax.legend(loc='best')
fig.tight_layout()
fig.savefig(OUT_DIR / 'V_T_afm_vs_T.png', dpi=300)
plt.show()

print('V_T^AFM = Phi_AFM (meV)')
for _, r in summary.iterrows():
    if np.isfinite(r['V_T_afm_c_meV']):
        print(f"  T={int(r['temperature_K']):>4d} K  c={r['V_T_afm_c_meV']:6.0f}  "
              f"b={r['V_T_afm_b_meV']:6.0f}")
'''

SEC7B_MD = (
    "### 7b. Calibration factor $c(T) = V_\\mathrm{peak}^\\mathrm{AFM}/V_T^\\mathrm{AFM}$\n\n"
    "The per-axis factor stays in a narrow band across the calibrated window, so the "
    "conversion $\\Phi = eV_\\mathrm{peak}/c$ is stable."
)

SEC7B_CODE = r'''fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
ax.plot(Tv[m_c], summary['c_calib_c'].values[m_c], 'o-',
        color=OKABE_ITO_CYCLE[2], label='c-axis')
ax.plot(Tv[m_b], summary['c_calib_b'].values[m_b], 's-',
        color=OKABE_ITO_CYCLE[4], mfc='white', label='b-axis')
ax.set_xlabel(r'$T$ (K)')
ax.set_ylabel(r'$c = V_\mathrm{peak}^\mathrm{AFM}/V_T^\mathrm{AFM}$')
ax.legend(loc='best')
fig.tight_layout()
fig.savefig(OUT_DIR / 'calibration_factor_vs_T.png', dpi=300)
plt.show()

print('calibration factor c')
for _, r in summary.iterrows():
    if np.isfinite(r['c_calib_c']):
        print(f"  T={int(r['temperature_K']):>4d} K  c-axis={r['c_calib_c']:.3f}  "
              f"b-axis={r['c_calib_b']:.3f}")
'''

SEC7C_MD = (
    "### 7c. Uncalibrated high-$T$ regime ($T > 80$ K)\n\n"
    "At 90 and 100 K the AFM FN minimum does not resolve, so no absolute barrier height "
    "is claimed. Only the relative conductance-peak splitting "
    "$\\Delta V_\\mathrm{peak} = V_\\mathrm{peak}^\\mathrm{AFM} - V_\\mathrm{peak}^\\mathrm{FM}$ "
    "is reported (in mV of bias, not energy)."
)

SEC7C_CODE = r'''hi = summary[~summary['calib_ok_c'].astype(bool)]
if len(hi):
    print('Uncalibrated regime (no AFM FN minimum) -- relative peak splitting only:')
    for _, r in hi.iterrows():
        dvp_c = r['Delta_ex_c_m2_meV']   # = raw dV_peak when calib_ok is False
        dvp_b = r['Delta_ex_b_m2_meV']
        print(f"  T={int(r['temperature_K']):>4d} K:  dV_peak(c) = {dvp_c:6.0f} mV,  "
              f"dV_peak(b) = {dvp_b:6.0f} mV   (V_T not resolved)")
else:
    print('All loaded temperatures calibrated (AFM V_T resolved).')
'''

SEC8_MD = (
    "## 8. Field-resolved Fowler-Nordheim plots at all temperatures\n\n"
    "$\\ln(|I|/V^2)$ vs $1/V$ for $|H_\\mathrm{z}|$ bins (AFM $\\to$ canted $\\to$ FM, "
    "coolwarm) at each temperature. The dashed vertical marks $1/V_T^\\mathrm{AFM}$ where "
    "the minimum resolves. The FN minimum is clean at low $T$ and is progressively lost "
    "above $\\sim 80$ K, which is exactly why 90 and 100 K cannot be calibrated."
)

SEC8_CODE = r'''from scripts.IV_Hscan_gaussian import load_dataframe
from scripts import fn_analysis as fn

DATA_DIR = PROJECT_ROOT / 'output' / 'IV_H_scans' / 'dataframes' / 'c_scans'
T_all = summary['temperature_K'].astype(int).tolist()

ncol = 3
nrow = int(np.ceil(len(T_all) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.2*ncol, 3.4*nrow), dpi=200,
                         sharex=True, sharey=True, squeeze=False)
cmap = plt.cm.coolwarm
for ax, T in zip(axes.ravel(), T_all):
    try:
        df = load_dataframe(DATA_DIR / f'IV_gaussian_{T}K.pkl')
    except FileNotFoundError:
        ax.axis('off')
        continue
    Hsat = float(summary.loc[summary['temperature_K'] == T, 'H_sat_c_T'].iloc[0])
    centers = list(np.round(np.linspace(0.0, round(Hsat, 2), 8), 2))
    cnorm = plt.Normalize(vmin=min(centers), vmax=max(centers))
    for rec in fn.field_binned_fn(df, centers, bin_hw=0.10):
        if rec['x'] is None:
            continue
        ax.plot(rec['x'], rec['y'], '-', color=cmap(cnorm(rec['H_center'])), lw=1.1)
    vt = summary.loc[summary['temperature_K'] == T, 'V_T_afm_c_meV'].iloc[0]
    if np.isfinite(vt):
        ax.axvline(1000.0 / vt, color='0.3', ls='--', lw=0.9, alpha=0.7)
    ax.text(0.96, 0.96, f'{T} K', transform=ax.transAxes, ha='right', va='top')
    ax.set_xlim(0, 20)
    ax.set_ylim(-18, -9)
for ax in axes.ravel()[len(T_all):]:
    ax.axis('off')
for ax in axes[-1, :]:
    ax.set_xlabel(r'$1/V$ (V$^{-1}$)')
for ax in axes[:, 0]:
    ax.set_ylabel(r'$\ln(|I|/V^2)$')
fig.tight_layout()
fig.savefig(OUT_DIR / 'FN_field_resolved_grid_all_T.png', dpi=200)
plt.show()
'''

SEC9_MD = (
    "## 9. AFM / FM FN endpoint overview vs $T$\n\n"
    "$\\ln(|I|/V^2)$ vs $1/V$ for the AFM ($|H_\\mathrm{z}| < 0.10$ T) and FM (saturated) "
    "endpoints, colour-coded by temperature. As $T$ rises the FN slope flattens and the "
    "magnetic-state separation collapses, the field-emission fingerprint of the "
    "$\\sim 80$ K crossover."
)

SEC9_CODE = r'''Tnorm = plt.Normalize(vmin=min(T_all), vmax=max(T_all))
cmapT = plt.cm.viridis

fig, (ax_a, ax_f) = plt.subplots(1, 2, figsize=(12, 5), dpi=300, sharey=True)
for T in T_all:
    try:
        df = load_dataframe(DATA_DIR / f'IV_gaussian_{T}K.pkl')
    except FileNotFoundError:
        continue
    V, IA, IF, nA, nF = fn.endpoint_curves(df, T)
    col = cmapT(Tnorm(T))
    for ax, I in [(ax_a, IA), (ax_f, IF)]:
        if I is None:
            continue
        x, yv, _ = fn.fn_transform(V, I)
        ax.plot(x, yv, '-', color=col, lw=1.4)
for ax, lab in [(ax_a, r'AFM ($|H_\mathrm{z}|<0.10$ T)'), (ax_f, 'FM (saturated)')]:
    ax.set_xlabel(r'$1/V$ (V$^{-1}$)')
    ax.set_xlim(0, 20)
    ax.set_ylim(-18, -9)
    ax.text(0.96, 0.04, lab, transform=ax.transAxes, ha='right', va='bottom',
            bbox=dict(facecolor='white', edgecolor='0.7', alpha=0.85,
                      boxstyle='round,pad=0.3'))
ax_a.set_ylabel(r'$\ln(|I|/V^2)$')
sm = plt.cm.ScalarMappable(norm=Tnorm, cmap=cmapT)
sm.set_array([])
cbar = fig.colorbar(sm, ax=[ax_a, ax_f])
cbar.set_label(r'$T$ (K)')
fig.savefig(OUT_DIR / 'FN_endpoint_overview_vs_T.png', dpi=300)
plt.show()
'''


def build():
    nb = nbformat.read(NB, as_version=4)
    cells = nb.cells

    # Refresh intro.
    cells[0].source = INTRO

    # Drop any previously inserted FN-summary cells (idempotent rebuild).
    cells[:] = [c for c in cells if TAG not in c.get("metadata", {}).get("tags", [])]

    # Insert before the "Save combined CSV" section.
    i_save = next(i for i, c in enumerate(cells)
                  if c.cell_type == "markdown" and "Save combined CSV" in c.source)
    new = [md(SEC7_MD), code(SEC7A_CODE),
           md(SEC7B_MD), code(SEC7B_CODE),
           md(SEC7C_MD), code(SEC7C_CODE),
           md(SEC8_MD), code(SEC8_CODE),
           md(SEC9_MD), code(SEC9_CODE)]
    for off, c in enumerate(new):
        cells.insert(i_save + off, c)

    nbformat.write(nb, NB)
    print(f"[written] {NB}")


if __name__ == "__main__":
    build()
