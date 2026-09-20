# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib", "scipy", "tqdm"]
# ///
"""
Regenerate Figure 1 panels c, d and e as standalone publication figures.

Panel c: bulk CrSBr SQUID magnetometry at 10 K, M/M_S along all three axes.
Panel d: junction current versus field along the hard c-axis (H_z).
Panel e: junction current versus field along the easy b-axis (H_y).

Panels d and e are the 20 K field sweeps with the current read off each
Gaussian-filtered I(V) curve at the probe bias V = +0.5 V, matching the
manuscript caption ("Magnetic-field sweeps with the junction current recorded
at 20 K and 0.5 V"). Panel c reuses the normalisation of the original SQUID
notebook: M_S is half the peak-to-peak moment of the loop.

The panels are written as separate PNGs at one figure size and the project RC
font sizes (labels 22, ticks 18), so all three scale together when tiled by
hand.

Colours follow the Figure 1 caption, which is the source of truth here:
a-axis orange, b-axis blue, c-axis bluish green. Panel d is a c-axis sweep and
so carries the same green as the c-axis curve in panel c; panel e the same blue.

Field-column note: in the stored device dataframes the signed sweep coordinate
is the column ``H``. The cryostat's Hy/Hz column names do not follow the
manuscript's axis labels (the c_scans sweep varies the column named Hy and vice
versa), so ``H`` is used directly and labelled per the manuscript convention.

Run:
    uv run scripts/make_fig1_panels.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.IV_Hscan_gaussian.dataframe_builder import get_current_at_voltage  # noqa: E402
from scripts.utils.notebook_setup import OKABE_ITO, configure_plot_style  # noqa: E402

V_PROBE = 0.5  # V, probe bias for the I(H) trace
TEMPERATURE = "20K"

DATAFRAMES = PROJECT_ROOT / "output" / "IV_H_scans" / "dataframes"
SQUID_DIR = PROJECT_ROOT / "data" / "SQUID_bulk_CrSBr" / "MvH"
OUTDIR = PROJECT_ROOT / "output" / "paper figures" / "fig1_panels"

COLOR_A_AXIS = OKABE_ITO["orange"]
COLOR_B_AXIS = OKABE_ITO["blue"]
COLOR_C_AXIS = OKABE_ITO["bluish_green"]

# (legend label, MPMS RSO file, saturation field in T quoted in the caption).
# The b-axis loop is the 1 T sweep: it saturates at the 0.3 T spin flip, so the
# 3 T file adds nothing but a longer flat tail.
SQUID_TRACES = [
    (r"$a$-axis", "CrSBr_Hmax_30000 Oe_IPhard_10K.rso.dat", 1.0),
    (r"$b$-axis", "CrSBr_Hmax_10000 Oe_IPeasy_10K.rso.dat", 0.3),
    (r"$c$-axis", "CrSBr_Hmax_30000 Oe_OOP_10K.rso.dat", 2.0),
]

SQUID_HEADER_ROWS = 30  # MPMS RSO preamble before the column header line


@dataclass
class SquidTrace:
    """One normalised bulk SQUID M(H) loop."""

    label: str
    field_T: np.ndarray
    M_norm: np.ndarray
    err_norm: np.ndarray
    sat_moment: float  # A m^2, half the peak-to-peak moment
    offset: float  # A m^2, loop midpoint


def load_squid_trace(filename: str, label: str) -> SquidTrace:
    """Read one MPMS RSO M(H) file and normalise it to M/M_S.

    M_S is half the peak-to-peak moment of the loop, as in the original
    SQUID_Bulk_MvsH notebook, so the normalised loop runs from -1 to +1.
    """
    data = pd.read_csv(SQUID_DIR / filename, delimiter=",", skiprows=SQUID_HEADER_ROWS)

    field_T = data["Field (Oe)"].to_numpy() / 10000  # Oe -> T
    moment = data["Long Moment (emu)"].to_numpy() / 1000  # emu -> A m^2
    moment_err = data["Long Scan Std Dev"].to_numpy() / 1000

    sat_moment = (np.max(moment) - np.min(moment)) / 2
    offset = (np.max(moment) + np.min(moment)) / 2

    return SquidTrace(
        label=label,
        field_T=field_T,
        M_norm=moment / sat_moment,
        err_norm=moment_err / sat_moment,
        sat_moment=float(sat_moment),
        offset=float(offset),
    )


def load_panel_trace(
    subfolder: str,
    v_probe: float = V_PROBE,
    temperature: str = TEMPERATURE,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (signed field in T, current in nA, actual probe bias in V).

    The trace is kept in acquisition order, so the field sweep direction and any
    hysteresis are preserved when it is plotted as a connected line.
    """
    df = pd.read_pickle(DATAFRAMES / subfolder / f"IV_gaussian_{temperature}.pkl")
    trace = get_current_at_voltage(df, v_probe, use_filtered=True)

    field = df.set_index("timestamp").loc[trace["timestamp"], "H"].to_numpy()
    current_nA = trace["I_at_V"].to_numpy() * 1e9
    v_actual = float(trace["V_actual"].iloc[0])

    return field, current_nA, v_actual


def make_squid_panel() -> Path:
    """Plot the three normalised bulk M(H) loops and save panel c."""
    traces = [load_squid_trace(filename, label) for label, filename, _ in SQUID_TRACES]
    colors = [COLOR_A_AXIS, COLOR_B_AXIS, COLOR_C_AXIS]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    for trace, color in zip(traces, colors):
        ax.errorbar(
            trace.field_T,
            trace.M_norm,
            trace.err_norm,
            fmt="-o",
            markersize=3,
            label=trace.label,
            color=color,
        )
    ax.set_xlabel(r"$H$ (T)")
    ax.set_ylabel(r"$M/M_\mathrm{S}$")
    ax.set_xticks(np.arange(-3, 4, 1))
    ax.legend(loc="best", frameon=False)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    outfile = OUTDIR / "fig1c_squid_MvsH_10K.png"
    fig.savefig(outfile, dpi=300)
    plt.close(fig)

    print(f"panel c -> {outfile.name}")
    for trace in traces:
        print(
            f"  {trace.label}: {len(trace.field_T)} points, "
            f"H = {trace.field_T.min():+.2f} to {trace.field_T.max():+.2f} T, "
            f"M_S = {trace.sat_moment:.3e} A m^2"
        )
    return outfile


def make_panel(panel: str, subfolder: str, xlabel: str, color: str) -> Path:
    """Plot I(H) at the probe bias for one field axis and save it."""
    field, current_nA, v_actual = load_panel_trace(subfolder)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    ax.plot(field, current_nA, "o-", markersize=4, linewidth=1.5, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$I$ (nA)")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    outfile = OUTDIR / f"fig1{panel}_{subfolder.split('_')[0]}_axis_20K.png"
    fig.savefig(outfile, dpi=300)
    plt.close(fig)

    print(
        f"panel {panel}: {len(field)} field steps, "
        f"H = {field.min():+.3f} to {field.max():+.3f} T, "
        f"I = {current_nA.min():.2f} to {current_nA.max():.2f} nA "
        f"at V = {v_actual:+.4f} V -> {outfile.name}"
    )
    return outfile


# (panel, dataframe subfolder, x label, colour)
PANELS = [
    ("d", "c_scans", r"$H_\mathrm{z}$ (T)", COLOR_C_AXIS),
    ("e", "b_scans", r"$H_\mathrm{y}$ (T)", COLOR_B_AXIS),
]


def main() -> None:
    configure_plot_style()
    make_squid_panel()
    for panel, subfolder, xlabel, color in PANELS:
        make_panel(panel, subfolder, xlabel, color)


if __name__ == "__main__":
    main()
