"""
H_scan Data Loader

Lightweight data reader for H_scan field sweep measurements from CrSBr experiments.
Reads tab-delimited .dat files containing magnetic field sweep data with lock-in
amplifier measurements.

File format:
- Header lines starting with '!' (metadata)
- Column header line
- Data rows (typically 82 steps per scan)

Usage:
    from scripts.Hscan.H_scan_dataloader import H_scan_dataloader

    loader = H_scan_dataloader()
    data = loader.load_file('path/to/datafile.dat')

    # Access data columns
    print(data.Hz_T)  # Hz component
    print(data.L1_Ch1)  # Lock-in 1 Channel 1 data
"""

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional


@dataclass
class H_scan_data:
    """
    Dataclass containing all columns from H_scan measurement files.

    Attributes:
        filename: Name of the source file
        date: Date strings from measurements
        time: Time strings from measurements
        Tcryo_K: Cryostat temperature in Kelvin
        Tsample_K: Sample temperature in Kelvin
        Hx_T: Magnetic field X-component in Tesla
        Hy_T: Magnetic field Y-component in Tesla
        Hz_T: Magnetic field Z-component in Tesla
        H_mag: Magnetic field magnitude |H| in Tesla
        phi: Phi angle in degrees
        theta: Theta angle in degrees
        L1_Ch1: Lock-in 1, Channel 1 (data)
        L1_Ch2: Lock-in 1, Channel 2 (error)
        L2_Ch1: Lock-in 2, Channel 1 (data)
        L2_Ch2: Lock-in 2, Channel 2 (error)
        L3_Ch1: Lock-in 3, Channel 1 (data)
        L3_Ch2: Lock-in 3, Channel 2 (error)
        L4_Ch1: Lock-in 4, Channel 1 (data)
        L4_Ch2: Lock-in 4, Channel 2 (error)
        L5_Ch1: Lock-in 5, Channel 1 (data)
        L5_Ch2: Lock-in 5, Channel 2 (error)
    """
    filename: str
    date: np.ndarray
    time: np.ndarray
    Tcryo_K: np.ndarray
    Tsample_K: np.ndarray
    Hx_T: np.ndarray
    Hy_T: np.ndarray
    Hz_T: np.ndarray
    H_mag: np.ndarray
    phi: np.ndarray
    theta: np.ndarray
    L1_Ch1: np.ndarray
    L1_Ch2: np.ndarray
    L2_Ch1: np.ndarray
    L2_Ch2: np.ndarray
    L3_Ch1: np.ndarray
    L3_Ch2: np.ndarray
    L4_Ch1: np.ndarray
    L4_Ch2: np.ndarray
    L5_Ch1: np.ndarray
    L5_Ch2: np.ndarray


class H_scan_dataloader:
    """
    Lightweight data loader for H_scan field sweep measurement files.

    Reads tab-delimited .dat files and returns all columns as numpy arrays
    in the H_scan_data dataclass.
    """

    def __init__(self):
        """Initialize the H_scan data loader."""
        pass

    def load_file(self, filepath: str) -> H_scan_data:
        """
        Load a single H_scan measurement file.

        Args:
            filepath: Path to the .dat file to load

        Returns:
            H_scan_data object containing all columns and filename

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is invalid
        """
        filepath = Path(filepath) # type: ignore

        if not filepath.exists(): # type: ignore
            raise FileNotFoundError(f"File not found: {filepath}")

        # Read the file and find where data starts
        df = self._read_data_file(filepath) # type: ignore

        # Extract filename
        filename = filepath.name # type: ignore

        # Create and return dataclass
        # Best Practice: Use .to_numpy() instead of .values
# This is more explicit and removes the FutureWarning.

        return H_scan_data(
            filename=filename,
            date=df['date'].to_numpy(),
            time=df['time'].to_numpy(),
            Tcryo_K=df['Tcryo (K)'].to_numpy(),
            Tsample_K=df['Tsample (K)'].to_numpy(),
            Hx_T=df['Hx (T)'].to_numpy(),
            Hy_T=df['Hy (T)'].to_numpy(),
            Hz_T=df['Hz (T)'].to_numpy(),
            H_mag=df['|H|'].to_numpy(),
            phi=df['phi'].to_numpy(),
            theta=df['theta'].to_numpy(),
            L1_Ch1=df['L1_Ch1'].to_numpy(),
            L1_Ch2=df['L1_Ch2'].to_numpy(),
            L2_Ch1=df['L2_Ch1'].to_numpy(),
            L2_Ch2=df['L2_Ch2'].to_numpy(),
            L3_Ch1=df['L3_Ch1'].to_numpy(),
            L3_Ch2=df['L3_Ch2'].to_numpy(),
            L4_Ch1=df['L4_Ch1'].to_numpy(),
            L4_Ch2=df['L4_Ch2'].to_numpy(),
            L5_Ch1=df['L5_ Ch1'].to_numpy(),
            L5_Ch2=df['L5_Ch2'].to_numpy()
        )


    def _read_data_file(self, filepath: Path) -> pd.DataFrame:
        """
        Read tab-delimited data file, skipping header lines.

        Args:
            filepath: Path to the data file

        Returns:
            pandas DataFrame with all columns
        """
        # Define column names explicitly to avoid issues with trailing tabs
        column_names = [
            'date', 'time', 'Tcryo (K)', 'Tsample (K)',
            'Hx (T)', 'Hy (T)', 'Hz (T)', '|H|', 'phi', 'theta',
            'L1_Ch1', 'L1_Ch2', 'L2_Ch1', 'L2_Ch2',
            'L3_Ch1', 'L3_Ch2', 'L4_Ch1', 'L4_Ch2',
            'L5_ Ch1', 'L5_Ch2'
        ]

        # Read file to find where data starts (skip lines beginning with '!' and empty lines)
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Find the column header line (first non-comment line with actual tab-separated content)
        header_line_idx = None
        for idx, line in enumerate(lines):
            stripped = line.strip()
            # Look for line that doesn't start with !, has content, and contains tabs
            if not stripped.startswith('!') and stripped and '\t' in stripped:
                header_line_idx = idx
                break

        if header_line_idx is None:
            raise ValueError(f"Could not find data header in file: {filepath}")

        # Read data using pandas, skipping header and using explicit column names
        df = pd.read_csv(
            filepath,
            sep='\t',
            skiprows=header_line_idx + 1,  # Skip to first data row (after column headers)
            names=column_names,  # Use explicit column names
            na_values=['NA', 'N/A', 'nan', 'NaN', ''],
            skipinitialspace=True,
            encoding='utf-8',
            usecols=range(20)  # Only read first 20 columns (ignore trailing empty columns)
        )

        # Remove any completely empty rows
        df = df.dropna(how='all', axis=0)

        return df


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        # Default example file
        filepath = str(__import__("pathlib").Path(__file__).resolve().parents[2] / "data" / "device 2" / "b_scans" / "H_scans" / "20251106153846b_scan_30K_92steps_H0.90T_phi_84.0to84.0_theta_0.0to0.0°_v0.dat")

    # Load data
    loader = H_scan_dataloader()
    data = loader.load_file(filepath)

    # Display information
    print(f"Loaded file: {data.filename}")
    print(f"Number of data points: {len(data.Hz_T)}")
    print(f"\nHz range: {data.Hz_T.min():.3f} to {data.Hz_T.max():.3f} T")
    print(f"Temperature: {data.Tcryo_K[0]:.1f} K")
    print(f"\nL1_Ch1 range: {data.L1_Ch1.min():.3e} to {data.L1_Ch1.max():.3e}")
    print(f"L2_Ch1 range: {data.L2_Ch1.min():.3e} to {data.L2_Ch1.max():.3e}")
    print(f"L3_Ch1 range: {data.L3_Ch1.min():.3e} to {data.L3_Ch1.max():.3e}")
    print(f"L4_Ch1 range: {data.L4_Ch1.min():.3e} to {data.L4_Ch1.max():.3e}")
