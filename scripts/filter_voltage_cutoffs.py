# ---
# description: |
#   Applies per-temperature voltage cutoffs to all TMR output files in
#   output/IV_H_scans/. Removes data rows beyond the physical measurement
#   range (compliance / short sweeps) from CSV and PKL files.
# entry_point: python scripts/filter_voltage_cutoffs.py
# dependencies:
#   - pandas
#   - numpy
# input: |
#   Hardcoded voltage_cutoffs dict at top of script (temperature in K -> max |V| in V).
#   Reads from output/IV_H_scans/dataframes/ and output/IV_H_scans/MR_summary/.
# process: |
#   For each temperature with a cutoff, filters rows where |Voltage (V)| > max_v
#   from per-temperature TMR CSV+PKL files and the combined summary file.
# output: |
#   Overwrites affected CSV and PKL files in-place. Prints a row-count
#   summary to stdout.
# last_updated: 2026-04-08
# ---

import pandas as pd
from pathlib import Path

# Voltage cutoffs per temperature (K: max |V| in V).
# Only temperatures listed here will have voltage limits applied.
voltage_cutoffs = {
    40:  0.75,
    110: 0.5,
    120: 0.5,
    130: 0.5,
    140: 0.5,
    150: 0.5,
    160: 0.5,
}

BASE = Path(__file__).parent.parent / "output" / "IV_H_scans"
DATAFRAMES = BASE / "dataframes"
MR_SUMMARY = BASE / "MR_summary"
AXES = ["b_scans", "c_scans"]


def apply_cutoff(df: pd.DataFrame, max_v: float) -> pd.DataFrame:
    return df[df["Voltage (V)"].abs() <= max_v].copy()


def filter_file(csv_path: Path, max_v: float, has_pkl: bool = True) -> None:
    if not csv_path.exists():
        print(f"  SKIP (not found): {csv_path.name}")
        return

    df = pd.read_csv(csv_path)
    before = len(df)
    df_filtered = apply_cutoff(df, max_v)
    after = len(df_filtered)
    removed = before - after

    df_filtered.to_csv(csv_path, index=False)
    if has_pkl:
        pkl_path = csv_path.with_suffix(".pkl")
        df_filtered.to_pickle(pkl_path)

    print(f"  {csv_path.name}: {before} -> {after} rows  ({removed} removed)")


def main() -> None:
    """Rewrite the TMR output files in place with the voltage cutoffs applied."""
    print("=" * 60)
    print("Applying voltage cutoffs — files will be overwritten in-place")
    print("=" * 60)

    # --- Per-temperature files ---
    for temp, max_v in sorted(voltage_cutoffs.items()):
        print(f"\n[{temp}K | max |V| = {max_v} V]")
        fname = f"TMR_ratio_vs_V_{temp}K.csv"

        for axis in AXES:
            # dataframes/ tier — CSV + PKL
            filter_file(DATAFRAMES / axis / fname, max_v, has_pkl=True)
            # MR_summary/ tier — CSV only
            filter_file(MR_SUMMARY / axis / fname, max_v, has_pkl=False)

    # --- Combined summary file ---
    print(f"\n[Combined: MR_summary_combined_all_temps]")
    combined_csv = MR_SUMMARY / "MR_summary_combined_all_temps.csv"
    combined_pkl = MR_SUMMARY / "MR_summary_combined_all_temps.pkl"

    if combined_csv.exists():
        df = pd.read_csv(combined_csv)
        before = len(df)

        mask = pd.Series(True, index=df.index)
        for temp, max_v in voltage_cutoffs.items():
            mask &= ~((df["temperature"] == temp) & (df["Voltage (V)"].abs() > max_v))

        df_filtered = df[mask].copy()
        after = len(df_filtered)
        removed = before - after

        df_filtered.to_csv(combined_csv, index=False)
        df_filtered.to_pickle(combined_pkl)
        print(f"  MR_summary_combined_all_temps.csv: {before} -> {after} rows  ({removed} removed)")
    else:
        print(f"  SKIP (not found): {combined_csv.name}")

    print("\nDone.")


# Guarded: this script overwrites data files, so importing the module
# (e.g. via autoreload in a notebook) must never trigger the rewrite.
if __name__ == "__main__":
    main()
