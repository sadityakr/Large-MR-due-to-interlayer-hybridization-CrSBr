"""Build a "Spectroscopy and FN analysis_{T}K" notebook from the existing
Barrier_vs_canting_{T}K notebook.

Reads the validated per-temperature spectroscopy notebook (shared-FWHM
Gaussian fit, Method-2 errors) and augments it in place-of-copy with:
the FN toolkit import, an FN section that measures V_T^AFM per axis and the
AFM-anchored calibration c = V_peak^AFM / V_T^AFM, and a rescale so the whole
downstream pipeline reports Phi = e*V_peak/c. Per-temperature inputs
(H_SAT_C, H_SF_B, overrides, spotcheck fields) are inherited unchanged from
the source notebook. The raw bias position is preserved in 'V_peak_bias' so
the spectroscopy overlay plots stay in real volts.

Input:
    --temperature T (K). Reads "Barrier_vs_canting_{T}K.ipynb" from the
    barrier-and-canting analysis folder.

Process:
    Locate cells by source substring and apply targeted edits / insertions
    via nbformat. Idempotent only in the sense that it always regenerates
    the output from the source.

Output:
    Writes "Spectroscopy and FN analysis_{T}K.ipynb" next to the source.
    Prints the output path.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = ["nbformat>=5.10"]
# ///
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat

NB_DIR = (Path(__file__).resolve().parents[1]
          / "notebooks" / "Device_2" / "barrier and canting analysis")


def md(text):
    return nbformat.v4.new_markdown_cell(text)


def code(text):
    return nbformat.v4.new_code_cell(text)


def find_cell(cells, needle, kind="code"):
    for i, c in enumerate(cells):
        if c.cell_type == kind and needle in c.source:
            return i
    raise ValueError(f"cell not found: {needle!r}")


def title_md(T):
    return (
        f"# {T} K spectroscopy and FN analysis (V5): AFM-anchored TVS calibration\n\n"
        "Combines the band-edge conductance-peak spectroscopy (shared-FWHM global "
        "Gaussian fit, Method-2 errors) with the Fowler-Nordheim (FN) analysis in one "
        "notebook, and applies the AFM-anchored transition-voltage calibration.\n\n"
        "The conductance-peak position $V_\\mathrm{peak}$ is a transport feature, not a "
        "barrier height. We measure the FN transition voltage $V_T$ (minimum of "
        "$\\ln(|I|/V^2)$ vs $1/V$) in the AFM endpoint, where the coherent interlayer "
        "channel is shut and the FN minimum is clean, and define the per-axis "
        "calibration $c = V_\\mathrm{peak}^\\mathrm{AFM}/V_T^\\mathrm{AFM}$. All barrier "
        "heights are then reported as $\\Phi = eV_\\mathrm{peak}/c$, so "
        "$\\Phi_\\mathrm{AFM} = eV_T^\\mathrm{AFM}$ by construction. Justification: "
        "`tmp/peer-review/TVS-barrier-calibration_thesis-section.md`.\n\n"
        f"Supersedes `Barrier_vs_canting_{T}K.ipynb` (which reported "
        "$\\Phi = eV_\\mathrm{peak}$, lever arm 1)."
    )


CALIB_MD = (
    "## 4e. Fowler-Nordheim transition voltage and per-axis calibration\n\n"
    "Measure $V_T^\\mathrm{AFM}$ per axis from the FN minimum and define "
    "$c = V_\\mathrm{peak}^\\mathrm{AFM}/V_T^\\mathrm{AFM}$. Every barrier height "
    "downstream is then $\\Phi = eV_\\mathrm{peak}/c$; the raw bias position is kept "
    "in `V_peak_bias` for the overlay plots. If the AFM $V_T$ does not resolve "
    "(higher $T$, FN regime lost), the notebook falls back to **uncalibrated** "
    "$V_\\mathrm{peak}$ and prints a warning."
)

CALIB_CODE = r'''# AFM-anchored transition-voltage calibration (per axis, this T).
def afm_endpoint_vpeak(res, h_afm_max):
    m = res['abs_H'] < h_afm_max
    mu, err, n = weighted_mean(res.loc[m, 'V_peak'], res.loc[m, 'V_peak_err'])
    return mu, err, n


def apply_calibration(res, c):
    """Phi = V_peak / c (eV). Preserve the raw bias position in 'V_peak_bias',
    then rescale V_peak / V_peak_err / Phi_* so the whole Method-2 pipeline
    operates on calibrated barrier heights and calibrated errors."""
    res = res.copy()
    if 'V_peak_bias' not in res.columns:
        res['V_peak_bias'] = res['V_peak'].astype(float)
    res['V_peak']     = res['V_peak_bias'] / c
    res['V_peak_err'] = res['V_peak_err'].astype(float) / c
    res['Phi_eV']     = res['V_peak']
    res['Phi_err_eV'] = res['V_peak_err']
    return res


# --- FN endpoints (per axis) -> V_T^AFM, B_FN ---
V_c, IA_c, IF_c, nA_c, nF_c = fn.endpoint_curves(df_c, TEMPERATURE,
                                                 h_afm_max=H_AFM_C_MAX, fm_min=H_FM_C_MIN)
V_b, IA_b, IF_b, nA_b, nF_b = fn.endpoint_curves(df_b, TEMPERATURE,
                                                 h_afm_max=H_AFM_B_MAX, fm_min=H_FM_B_MIN)
fn_afm_c, fn_fm_c = fn.fit_fn(V_c, IA_c), fn.fit_fn(V_c, IF_c)
fn_afm_b, fn_fm_b = fn.fit_fn(V_b, IA_b), fn.fit_fn(V_b, IF_b)
V_T_afm_c = fn_afm_c['V_T'] if fn_afm_c else float('nan')
V_T_afm_b = fn_afm_b['V_T'] if fn_afm_b else float('nan')

# --- V_peak^AFM (raw) and per-axis calibration factor ---
vpk_afm_c, _, _ = afm_endpoint_vpeak(res_c, H_AFM_C_MAX)
vpk_afm_b, _, _ = afm_endpoint_vpeak(res_b, H_AFM_B_MAX)
c_c = fn.calibration_factor(vpk_afm_c, V_T_afm_c)
c_b = fn.calibration_factor(vpk_afm_b, V_T_afm_b)
CALIB_OK_C, CALIB_OK_B = bool(np.isfinite(c_c)), bool(np.isfinite(c_b))

if CALIB_OK_C:
    res_c = apply_calibration(res_c, c_c)
else:
    print('WARNING: c-axis AFM V_T did not resolve; Phi is UNCALIBRATED V_peak.')
    res_c['V_peak_bias'] = res_c['V_peak']
    res_c['Phi_eV'] = res_c['V_peak']; res_c['Phi_err_eV'] = res_c['V_peak_err']
if CALIB_OK_B:
    res_b = apply_calibration(res_b, c_b)
else:
    print('WARNING: b-axis AFM V_T did not resolve; Phi is UNCALIBRATED V_peak.')
    res_b['V_peak_bias'] = res_b['V_peak']
    res_b['Phi_eV'] = res_b['V_peak']; res_b['Phi_err_eV'] = res_b['V_peak_err']

for ax, vt, c_, vpk, BA, BF in [('c', V_T_afm_c, c_c, vpk_afm_c, fn_afm_c, fn_fm_c),
                                ('b', V_T_afm_b, c_b, vpk_afm_b, fn_afm_b, fn_fm_b)]:
    bA = BA['B_FN'] if BA else float('nan')
    bF = BF['B_FN'] if BF else float('nan')
    vt_str = f'{vt*1000:.0f} meV' if np.isfinite(vt) else 'nan (no FN minimum)'
    print(f'{ax}-axis: V_peak^AFM = {vpk*1000:.0f} mV, V_T^AFM = {vt_str}, '
          f'c = {c_:.3f},  Phi_AFM = e*V_T = {vt_str}  '
          f'(B_FN^AFM={bA:.3f} V, B_FN^FM={bF:.3f} V)')
'''

FN_ENDPOINT_MD = (
    "## 4f. FN regime identification: AFM vs FM endpoints (c-axis)\n\n"
    "$\\ln(|I|/V^2)$ vs $1/V$ for the AFM and FM endpoints. The AFM curve shows the "
    "clean transition-voltage minimum $V_T^\\mathrm{AFM}$ that anchors the calibration; "
    "the FM curve has no FN minimum (the coherent interlayer channel fills in the knee). "
    "Dashed verticals mark $1/V_T$."
)

FN_ENDPOINT_CODE = r'''fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
for label, fr, dcol, fcol, mk in [
        ('AFM', fn_afm_c, OKABE_ITO_CYCLE[1], OKABE_ITO_CYCLE[6], 'o'),
        ('FM',  fn_fm_c,  OKABE_ITO_CYCLE[5], OKABE_ITO_CYCLE[3], 's')]:
    if fr is None:
        continue
    ax.plot(fr['x'], fr['y'], mk, color=dcol, ms=3, alpha=0.5, label=f'{label} data')
    i0, i1 = fr['i_window']
    xs = fr['x'][i0:i1]
    ax.plot(xs, fr['slope'] * xs + fr['intercept'], ls=':', color=fcol, lw=2.4,
            label=f'{label} FN fit')
    if np.isfinite(fr['V_T']):
        ax.axvline(1.0 / fr['V_T'], color=fcol, ls='--', lw=1.0, alpha=0.7)
ax.set_xlabel(r'$1/V$ (V$^{-1}$)')
ax.set_ylabel(r'$\ln(|I|/V^2)$')
ax.legend(loc='lower left')
fig.tight_layout()
fig.savefig(OUT_C / f'FN_endpoints_{TEMPERATURE}K_c.png', dpi=300)
plt.show()

for label, fr in [('AFM', fn_afm_c), ('FM', fn_fm_c)]:
    if fr is None:
        print(f'  {label}: no FN regime')
        continue
    vt = fr['V_T']
    vt_str = f'{vt*1000:.0f} meV' if np.isfinite(vt) else 'nan'
    print(f"  {label}: B_FN = {fr['B_FN']:.3f} V, V_T = {vt_str}, R^2 = {fr['R2']:.4f}")
'''

FN_FIELD_MD = (
    "## 4g. Field-resolved FN plot (c-axis canting trajectory)\n\n"
    "$\\ln(|I|/V^2)$ vs $1/V$ for $|H_\\mathrm{z}|$ bins from AFM ($H\\approx 0$) "
    "through the canted states to saturated FM. This is the field-axis counterpart of "
    "the FN_five_points figure: the FN trace rotates and translates smoothly between "
    "the AFM and FM endpoints as the canting angle grows."
)

FN_FIELD_CODE = r'''H_BIN_CENTERS = [0.00, 0.30, 0.60, 0.90, 1.20, 1.50, 1.80, round(H_SAT_C, 2)]
fn_bins = fn.field_binned_fn(df_c, H_BIN_CENTERS, bin_hw=0.10)

cnorm = plt.Normalize(vmin=min(H_BIN_CENTERS), vmax=max(H_BIN_CENTERS))
cmap = plt.cm.coolwarm   # blue (AFM, low H) -> red (FM, high H)
fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
for rec in fn_bins:
    if rec['x'] is None:
        continue
    col = cmap(cnorm(rec['H_center']))
    ax.plot(rec['x'], rec['y'], '-', color=col, lw=1.5)
    fr = rec['fit']
    if fr is not None and (rec['H_center'] == 0.0 or rec['H_center'] >= H_FM_C_MIN):
        i0, i1 = fr['i_window']
        xs = fr['x'][i0:i1]
        ax.plot(xs, fr['slope'] * xs + fr['intercept'], ls=':', color=col, lw=2.0)
if np.isfinite(V_T_afm_c):
    ax.axvline(1.0 / V_T_afm_c, color='0.3', ls='--', lw=1.0, alpha=0.7)
sm = plt.cm.ScalarMappable(norm=cnorm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label(r'$|H_\mathrm{z}|$ (T)')
ax.set_xlabel(r'$1/V$ (V$^{-1}$)')
ax.set_ylabel(r'$\ln(|I|/V^2)$')
fig.tight_layout()
fig.savefig(OUT_C / f'FN_field_resolved_{TEMPERATURE}K_c.png', dpi=300)
plt.show()

print(f"{'H_center (T)':>13}  {'N':>5}  {'B_FN (V)':>10}  {'V_T (meV)':>10}")
for rec in fn_bins:
    B, VT = rec['B_FN'], rec['V_T']
    Bs = f'{B:10.3f}' if np.isfinite(B) else f"{'nan':>10}"
    VTs = f'{VT*1000:10.0f}' if np.isfinite(VT) else f"{'nan':>10}"
    print(f"{rec['H_center']:13.2f}  {rec['n_curves']:5d}  {Bs}  {VTs}")
'''

CALIB_RECAP_CODE = r'''# FN calibration factor recap (used for the field-dependent Phi below).
# Phi = e * V_peak / c, with c = V_peak^AFM / V_T^AFM measured per axis in 4e.
for ax_lbl, vpk, vt, c_, ok in [('c', vpk_afm_c, V_T_afm_c, c_c, CALIB_OK_C),
                                ('b', vpk_afm_b, V_T_afm_b, c_b, CALIB_OK_B)]:
    if ok:
        print(f'{ax_lbl}-axis:  c = V_peak^AFM / V_T^AFM = '
              f'{vpk*1000:.0f} mV / {vt*1000:.0f} mV = {c_:.3f}'
              f'   ->  Phi_AFM = e*V_T^AFM = {vt*1000:.0f} meV')
    else:
        print(f'{ax_lbl}-axis:  V_T^AFM unresolved -> Phi reported UNCALIBRATED')
'''

COS_FIT_CELL = r'''# --- Below-saturation slice; both fits + standalone cos(theta/2) publication plot. ---
def weighted_linear_fit(x, y, yerr):
    """Inverse-variance weighted fit y = a + b*x. Returns dict with a, b,
    a_err, b_err, cov_ab, chi2, chi2_red, dof, n, birge_scale."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)
    pos = yerr > 0
    safe = np.where(pos, yerr, np.median(yerr[pos]) if pos.any() else 1.0)
    w = 1.0 / safe**2
    S, Sx, Sxx = w.sum(), (w*x).sum(), (w*x*x).sum()
    Sy, Sxy = (w*y).sum(), (w*x*y).sum()
    det = S*Sxx - Sx*Sx
    a = (Sxx*Sy - Sx*Sxy) / det
    b = (S*Sxy - Sx*Sy) / det
    var_a, var_b, cov_ab = Sxx/det, S/det, -Sx/det
    resid = y - (a + b*x)
    chi2 = float(np.sum(w * resid**2))
    dof = max(int(x.size - 2), 1)
    chi2_red = chi2 / dof
    scale = max(1.0, chi2_red)
    var_a, var_b, cov_ab = var_a*scale, var_b*scale, cov_ab*scale
    return dict(a=float(a), b=float(b),
                a_err=float(np.sqrt(var_a)), b_err=float(np.sqrt(var_b)),
                cov_ab=float(cov_ab), chi2=chi2, chi2_red=float(chi2_red),
                dof=dof, n=int(x.size), birge_scale=float(scale))


below_sat = (agg_c_signed['abs_H'].values < H_SAT_C) & np.isfinite(agg_c_signed['Phi_err_m2_eV'].values)
sub = agg_c_signed.loc[below_sat].copy()
sub['cos_half'] = sub['abs_H'].values / H_SAT_C
sub['sin2_half'] = 1.0 - (sub['abs_H'].values / H_SAT_C)**2

x_cos = sub['cos_half'].values
x_sin2 = sub['sin2_half'].values
y = sub['Phi_eV'].values
yerr = sub['Phi_err_m2_eV'].values

fit_lin = weighted_linear_fit(x_cos, y, yerr)
fit_sin2 = weighted_linear_fit(x_sin2, y, yerr)

# Standalone cos(theta/2) fit (publication figure: no title).
fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
xx = np.linspace(0.0, 1.0, 200)
yy = fit_lin['a'] + fit_lin['b'] * xx
band = np.sqrt(fit_lin['a_err']**2 + 2*xx*fit_lin['cov_ab'] + xx**2 * fit_lin['b_err']**2)
ax.errorbar(x_cos, 1000*y, yerr=1000*yerr, fmt='o', color=OKABE_ITO_CYCLE[2],
            ms=4, alpha=0.9, ecolor=OKABE_ITO_CYCLE[2], elinewidth=0.8,
            capsize=2, label='Data')
ax.plot(xx, 1000*yy, '-', color='black', lw=1.3, label='Linear fit')
ax.fill_between(xx, 1000*(yy-band), 1000*(yy+band), color='black', alpha=0.12, lw=0)
ax.set_xlabel(r'$\cos(\theta/2) = |H_\mathrm{z}| / H_\mathrm{sat}^\mathrm{c}$')
ax.set_ylabel(r'$\Phi$ (meV)')
ax.set_xlim(-0.05, 1.05)
ax.legend(loc='best')
fig.tight_layout()
fig.savefig(OUT_C / f'Phi_vs_cos_half_{TEMPERATURE}K_linear_fit.png', dpi=300)
plt.show()

print(f'[linear  Phi = a + b*cos(theta/2), |H_z| < {H_SAT_C} T]')
print(f'  a       = ({1000*fit_lin["a"]:.2f} +/- {1000*fit_lin["a_err"]:.2f}) meV   [Phi at cos(theta/2)=0]')
print(f'  b       = ({1000*fit_lin["b"]:.2f} +/- {1000*fit_lin["b_err"]:.2f}) meV   [slope per unit cos(theta/2)]')
print(f'  chi^2   = {fit_lin["chi2"]:.2f}  chi^2_red = {fit_lin["chi2_red"]:.2f}  dof = {fit_lin["dof"]}  n = {fit_lin["n"]}')
'''

SIN2_SCALARS_CELL = r'''# --- sin^2(theta/2) endpoint values from fit_sin2 (for the summary CSV). ---
Phi_FM_sin2 = fit_sin2['a']
Phi_FM_sin2_err = fit_sin2['a_err']
Phi_AFM_sin2 = fit_sin2['a'] + fit_sin2['b']
Phi_AFM_sin2_err = float(np.sqrt(fit_sin2['a_err']**2 + fit_sin2['b_err']**2 + 2*fit_sin2['cov_ab']))
Delta_sin2 = fit_sin2['b']
Delta_sin2_err = fit_sin2['b_err']

print('[sin^2(theta/2)  Phi = a + b*sin^2(theta/2)]')
print(f'  Phi_FM  = ({1000*Phi_FM_sin2:.2f} +/- {1000*Phi_FM_sin2_err:.2f}) meV')
print(f'  Phi_AFM = ({1000*Phi_AFM_sin2:.2f} +/- {1000*Phi_AFM_sin2_err:.2f}) meV')
print(f'  Delta   = ({1000*Delta_sin2:.2f} +/- {1000*Delta_sin2_err:.2f}) meV')
print(f'  chi^2   = {fit_sin2["chi2"]:.2f}  chi^2_red = {fit_sin2["chi2_red"]:.2f}  dof = {fit_sin2["dof"]}  n = {fit_sin2["n"]}')
'''

COMPARISON_MD = (
    "### Fit-model comparison (supplementary)\n\n"
    "The same below-saturation $\\Phi(H_\\mathrm{z})$ fitted with $\\cos(\\theta/2)$ "
    "(left) and $\\sin^2(\\theta/2)$ (right), with $\\chi^2$ in each panel title. The "
    "linear $\\cos(\\theta/2)$ law is the better description of the canting dependence."
)

COMPARISON_CELL = r'''# --- Side-by-side fit comparison (supplementary): chi^2 in each panel title. ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300, sharey=True)
for ax, xv, fit, xlabel, fitlabel in [
        (axes[0], x_cos, fit_lin,
         r'$\cos(\theta/2) = |H_\mathrm{z}| / H_\mathrm{sat}^\mathrm{c}$', 'Linear fit'),
        (axes[1], x_sin2, fit_sin2,
         r'$\sin^2(\theta/2) = 1 - (H_\mathrm{z}/H_\mathrm{sat}^\mathrm{c})^2$',
         r'$\sin^2(\theta/2)$ fit')]:
    xx = np.linspace(0.0, 1.0, 200)
    yy = fit['a'] + fit['b'] * xx
    band = np.sqrt(fit['a_err']**2 + 2*xx*fit['cov_ab'] + xx**2 * fit['b_err']**2)
    ax.errorbar(xv, 1000*y, yerr=1000*yerr, fmt='o', color=OKABE_ITO_CYCLE[2],
                ms=4, alpha=0.9, ecolor=OKABE_ITO_CYCLE[2], elinewidth=0.8,
                capsize=2, label='Data')
    ax.plot(xx, 1000*yy, '-', color='black', lw=1.3, label=fitlabel)
    ax.fill_between(xx, 1000*(yy-band), 1000*(yy+band), color='black', alpha=0.12, lw=0)
    ax.set_xlabel(xlabel)
    ax.set_xlim(-0.05, 1.05)
    ax.set_title(rf'$\chi^2 = {fit["chi2"]:.1f}$,  $\chi^2_\mathrm{{red}} = {fit["chi2_red"]:.2f}$')
    ax.legend(loc='best')
axes[0].set_ylabel(r'$\Phi$ (meV)')
fig.tight_layout()
fig.savefig(OUT_C / f'Phi_fit_comparison_{TEMPERATURE}K.png', dpi=300)
plt.show()
'''

SUMMARY_OLD = ("pd.DataFrame([summary_row]).to_csv(OUT_C / "
               "f'fit_summary_{TEMPERATURE}K.csv', index=False)")
SUMMARY_NEW = r'''summary_row.update({
    'c_calib_c': c_c, 'c_calib_b': c_b,
    'V_T_afm_c_meV': 1000 * V_T_afm_c, 'V_T_afm_b_meV': 1000 * V_T_afm_b,
    'B_FN_afm_c_V': (fn_afm_c['B_FN'] if fn_afm_c else float('nan')),
    'B_FN_fm_c_V':  (fn_fm_c['B_FN']  if fn_fm_c  else float('nan')),
    'calib_ok_c': CALIB_OK_C, 'calib_ok_b': CALIB_OK_B,
})
pd.DataFrame([summary_row]).to_csv(OUT_C / f'fit_summary_{TEMPERATURE}K.csv', index=False)'''


def build(T):
    src_path = NB_DIR / f"Barrier_vs_canting_{T}K.ipynb"
    out_path = NB_DIR / f"Spectroscopy and FN analysis_{T}K.ipynb"
    nb = nbformat.read(src_path, as_version=4)
    cells = nb.cells

    # 1. Title.
    cells[0].source = title_md(T)

    # 2. Imports: add the FN module.
    i_imp = find_cell(cells, "from scripts.fit_cache import load_or_compute_shared_fit")
    if "import fn_analysis" not in cells[i_imp].source:
        cells[i_imp].source += "\nfrom scripts import fn_analysis as fn"

    # 3. Insert FN + calibration cells right after res_c/res_b are built.
    i_res = find_cell(cells, "res_c = add_measured_fwhm(None, curves_c, fit_c['df'])")
    new_cells = [md(CALIB_MD), code(CALIB_CODE),
                 md(FN_FIELD_MD), code(FN_FIELD_CODE)]
    for off, c in enumerate(new_cells, start=1):
        cells.insert(i_res + off, c)

    # 4. Spectroscopy plots must mark the raw bias position, not calibrated Phi.
    for needle in ["spotcheck_panel", "def plot_fit_overview"]:
        i = find_cell(cells, needle)
        cells[i].source = cells[i].source.replace(
            "ax.axvline(row['V_peak']", "ax.axvline(row['V_peak_bias']")

    # 5. Override branch: recalibrate after any manual refit (no-op if empty).
    i_ovr = find_cell(cells, "OVERRIDES_C = {")
    s = cells[i_ovr].source
    s = s.replace(
        "    res_c['Phi_eV']         = res_c['V_peak']\n"
        "    res_c['Phi_err_eV']     = res_c['V_peak_err']\n"
        "    res_c, sigma_floor_c, n_fm_c = inflate_errors(res_c, H_FM_C_MIN_EFF)",
        "    res_c = apply_calibration(res_c, c_c) if CALIB_OK_C else res_c\n"
        "    res_c, sigma_floor_c, n_fm_c = _setup_method2_columns(res_c, H_FM_C_MIN_EFF)")
    s = s.replace(
        "    res_b['Phi_eV']         = res_b['V_peak']\n"
        "    res_b['Phi_err_eV']     = res_b['V_peak_err']\n"
        "    res_b, sigma_floor_b, n_fm_b = inflate_errors(res_b, H_FM_B_MIN_EFF)",
        "    res_b = apply_calibration(res_b, c_b) if CALIB_OK_B else res_b\n"
        "    res_b, sigma_floor_b, n_fm_b = _setup_method2_columns(res_b, H_FM_B_MIN_EFF)")
    cells[i_ovr].source = s

    # 6. Show the calibration factor again right before the field-dependent
    #    Phi(H) plot (section 7).
    i_sec7 = find_cell(cells, "## 7. Canonical", kind="markdown")
    cells.insert(i_sec7, code(CALIB_RECAP_CODE))

    # 7. Legend font size is set centrally (rcParams = 15); drop per-call
    #    fontsize=16 overrides so the central value applies.
    for c in cells:
        if c.cell_type == "code" and "fontsize=16" in c.source:
            c.source = c.source.replace(', fontsize=16)', ')')

    # 7b. b-axis legend: drop the "(excluded)" qualifier.
    for c in cells:
        if c.cell_type == "code" and "spin-flip (excluded)" in c.source:
            c.source = c.source.replace("spin-flip (excluded)", "spin-flip")

    # 8. Section 8: keep the standalone cos(theta/2) publication plot, add a
    #    side-by-side cos vs sin^2 comparison (chi^2 in each title) for the
    #    supplementary, and reduce the sin^2 cell to its summary scalars.
    i_cos = find_cell(cells, "def weighted_linear_fit")
    cells[i_cos].source = COS_FIT_CELL
    cells.insert(i_cos + 1, code(COMPARISON_CELL))
    cells.insert(i_cos + 1, md(COMPARISON_MD))
    i_sin2 = find_cell(cells, "Phi_FM_sin2      = fit_sin2['a']")
    cells[i_sin2].source = SIN2_SCALARS_CELL

    # 9. Persist the calibration in the summary CSV.
    i_sum = find_cell(cells, SUMMARY_OLD)
    cells[i_sum].source = cells[i_sum].source.replace(SUMMARY_OLD, SUMMARY_NEW)

    # Clear stale outputs / execution counts, and collapse code input by
    # default (markdown + outputs stay visible; click to expand).
    for c in cells:
        if c.cell_type == "code":
            c.outputs = []
            c.execution_count = None
            c.metadata.setdefault("jupyter", {})["source_hidden"] = True

    nbformat.write(nb, out_path)
    print(f"[written] {out_path}")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--temperature", "-T", type=int, default=20,
                    help="temperature in K; reads Barrier_vs_canting_{T}K.ipynb")
    args = ap.parse_args()
    build(args.temperature)
