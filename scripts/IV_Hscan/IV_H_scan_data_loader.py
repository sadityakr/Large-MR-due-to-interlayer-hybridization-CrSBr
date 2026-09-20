"""
IV Data Loader
Created: 05/11/2025

Lightweight data reader for IV measurement files.
Reads tab-delimited text files containing voltage and current data.
"""

import numpy as np
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class IVData:
    """
    Data container for IV measurements.

    Attributes:
        filename: Name of the file (without path)
        filepath: Full path to the data file
        voltage: Array of voltage values (V)
        current: Array of current values (A)
    """
    filename: str
    filepath: str
    voltage: np.ndarray
    current: np.ndarray


@dataclass
class IVDataExtended:
    """
    Extended data container for IV measurements with parsed metadata.

    Attributes:
        filename: Name of the file (without path)
        filepath: Full path to the data file
        voltage: Array of voltage values (V)
        current: Array of current values (A)
        Hx: Magnetic field x-component (T)
        Hy: Magnetic field y-component (T)
        Hz: Magnetic field z-component (T)
        H: Total magnetic field magnitude with sign (T)
           Sign determined by component with larger magnitude (Hx or Hy)
        timestamp: Measurement timestamp (YYYYMMDDHHMMSS)
    """
    filename: str
    filepath: str
    voltage: np.ndarray
    current: np.ndarray
    Hx: float
    Hy: float
    Hz: float
    H: float
    timestamp: str


class IVDataLoader:
    """
    Lightweight loader for IV measurement data files.

    Reads tab-delimited text files with headers:
    'Voltage (V)' and 'Current (A)'
    """

    def __init__(self):
        pass

    def load(self, filepath: str) -> IVData:
        """
        Load IV data from a file.

        Parameters:
            filepath: Path to the IV data file

        Returns:
            IVData object containing filename, filepath, and data arrays

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is incorrect
        """
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        try:
            # Load data, skip header row
            data = np.loadtxt(filepath, skiprows=1, delimiter='\t')

            # Extract voltage and current columns
            voltage = data[:, 0]
            current = data[:, 1]

            return IVData(
                filename=path.name,
                filepath=str(path.absolute()),
                voltage=voltage,
                current=current
            )

        except Exception as e:
            raise ValueError(f"Error loading IV data from {filepath}: {str(e)}")


def load_iv_data(filepath: str) -> IVData:
    """
    Convenience function to load a single IV data file.

    Parameters:
        filepath: Path to the IV data file

    Returns:
        IVData object containing the data
    """
    loader = IVDataLoader()
    return loader.load(filepath)


def data_parser_IV(iv_data: IVData) -> IVDataExtended:
    """
    Parse filename to extract magnetic field components and timestamp.

    Expects filename format:
    YYYYMMDDHHMMSS_Hx_VALUE_T_Hy_VALUE_T_Hz_VALUE_T_*.dat

    Parameters:
        iv_data: IVData object with filename to parse

    Returns:
        IVDataExtended object with parsed Hx, Hy, Hz, H, and timestamp

    Raises:
        ValueError: If filename cannot be parsed
    """
    filename = iv_data.filename

    # Extract timestamp (14 digits at start)
    timestamp_match = re.match(r'^(\d{14})_', filename)
    if not timestamp_match:
        raise ValueError(f"Cannot extract timestamp from filename: {filename}")
    timestamp = timestamp_match.group(1)

    # Extract Hx value
    hx_match = re.search(r'Hx_([-+]?\d+\.\d+E[+-]?\d+)_T', filename)
    if not hx_match:
        raise ValueError(f"Cannot extract Hx from filename: {filename}")
    Hx = float(hx_match.group(1))

    # Extract Hy value
    hy_match = re.search(r'Hy_([-+]?\d+\.\d+E[+-]?\d+)_T', filename)
    if not hy_match:
        raise ValueError(f"Cannot extract Hy from filename: {filename}")
    Hy = float(hy_match.group(1))

    # Extract Hz value
    hz_match = re.search(r'Hz_([-+]?\d+\.\d+E[+-]?\d+)_T', filename)
    if not hz_match:
        raise ValueError(f"Cannot extract Hz from filename: {filename}")
    Hz = float(hz_match.group(1))

    # Calculate total field magnitude
    H_magnitude = np.sqrt(Hx**2 + Hy**2 + Hz**2)

    # Determine sign based on component with larger magnitude
    if abs(Hx) >= abs(Hy):
        H_sign = np.sign(Hx)
    else:
        H_sign = np.sign(Hy)

    H = H_magnitude * H_sign

    return IVDataExtended(
        filename=iv_data.filename,
        filepath=iv_data.filepath,
        voltage=iv_data.voltage,
        current=iv_data.current,
        Hx=Hx,
        Hy=Hy,
        Hz=Hz,
        H=H,
        timestamp=timestamp
    )


# Example usage
if __name__ == "__main__":
    # Example file path
    example_file = r"data/device 2/20251104191228TMR_CBS_10K_w_IV_82steps_H0.00T_phi_84.0to84.0_theta_0.0to0.0°_v0.datfolder/20251104193843_Hx_41.820000E-3_T_Hy_-150.000000E-6_T_Hz_397.900000E-3_T_decreasing.txt.dat"

    # Load data
    data = load_iv_data(example_file)

    # Display basic info
    print("=== Basic IV Data ===")
    print(f"Filename: {data.filename}")
    print(f"Voltage range: {data.voltage.min():.3f} to {data.voltage.max():.3f} V")
    print(f"Current range: {data.current.min():.3e} to {data.current.max():.3e} A")
    print(f"Number of data points: {len(data.voltage)}")

    # Parse metadata from filename
    print("\n=== Extended IV Data with Parsed Metadata ===")
    extended_data = data_parser_IV(data)
    print(f"Timestamp: {extended_data.timestamp}")
    print(f"Hx: {extended_data.Hx:.6e} T")
    print(f"Hy: {extended_data.Hy:.6e} T")
    print(f"Hz: {extended_data.Hz:.6e} T")
    print(f"H (signed magnitude): {extended_data.H:.6e} T")
    print(f"  |Hx| = {abs(extended_data.Hx):.6e} T, |Hy| = {abs(extended_data.Hy):.6e} T")
    print(f"  Sign determined by: {'Hx' if abs(extended_data.Hx) >= abs(extended_data.Hy) else 'Hy'}")
