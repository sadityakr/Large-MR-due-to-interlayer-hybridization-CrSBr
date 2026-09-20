"""
DataFrame builder module for IV H-scan Gaussian filter analysis results.

This module converts filtered IV data into pandas DataFrames and provides
utilities for data export and retrieval.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Union, Optional

from .IV_gaussian_filter import IVFilteredData


def build_dataframe(filtered_dict: Dict[str, IVFilteredData],
                   temperature: Optional[float] = None) -> pd.DataFrame:
    """
    Convert filtered IV data dictionary to pandas DataFrame.

    This function preserves RAW voltage/current arrays for verification
    alongside filtered curves and quality metrics.

    Parameters:
        filtered_dict: Dictionary mapping timestamp → IVFilteredData
        temperature: Optional temperature value (K) to add to all rows

    Returns:
        pandas DataFrame with columns:
            Metadata: timestamp, filename, H, Hx, Hy, Hz, temperature
            Filter params: sigma, v_offset, residual_rms, n_outliers_removed
            Raw data: voltage (array), current (array)
            Filtered data: voltage_smooth (array), current_filtered (array), current_smooth (array)
    """
    rows = []

    for timestamp, data in filtered_dict.items():
        row = {
            # Metadata
            'timestamp': data.timestamp,
            'filename': data.filename,
            'H': data.H,
            'Hx': data.Hx,
            'Hy': data.Hy,
            'Hz': data.Hz,

            # Filter parameters and results
            'sigma': data.sigma,
            'v_offset': data.v_offset,
            'residual_rms': data.residual_rms,
            'n_outliers_removed': data.n_outliers_removed,

            # RAW DATA (for verification and debugging)
            'voltage': data.voltage,               # Original measured voltages
            'current': data.current,               # Original measured currents

            # FILTERED DATA
            'current_filtered': data.current_filtered,  # Filtered at original V points
            'voltage_smooth': data.voltage_smooth,      # Smooth voltage array (500 pts)
            'current_smooth': data.current_smooth,      # Smooth filtered current (500 pts)

            # SYMMETRIC/ASYMMETRIC DECOMPOSITION
            'current_filtered_sym': data.current_filtered_sym,    # Symmetric component (original V)
            'current_filtered_asym': data.current_filtered_asym,  # Asymmetric component (original V)
            'current_smooth_sym': data.current_smooth_sym,        # Smooth symmetric (500 pts)
            'current_smooth_asym': data.current_smooth_asym,      # Smooth asymmetric (500 pts)
        }

        # Add temperature if provided
        if temperature is not None:
            row['temperature'] = temperature

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by timestamp to preserve hysteresis order
    df = df.sort_values('timestamp').reset_index(drop=True)

    return df


def save_dataframe(df: pd.DataFrame,
                   output_path: Union[str, Path],
                   format: str = 'pickle',
                   save_csv_scalars: bool = True) -> None:
    """
    Save DataFrame to file.

    Parameters:
        df: DataFrame to save
        output_path: Output file path (extension will be added automatically)
        format: Save format - 'pickle' (preserves arrays) or 'hdf5'
        save_csv_scalars: Also save scalar columns to CSV for spreadsheet viewing

    Notes:
        - Pickle format preserves numpy arrays in DataFrame cells
        - CSV export excludes array columns (voltage, current, etc.)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == 'pickle':
        # Save as pickle (preserves all data types)
        pickle_path = output_path.with_suffix('.pkl')
        df.to_pickle(pickle_path)
        print(f"DataFrame saved to: {pickle_path}")

    elif format == 'hdf5':
        # Save as HDF5 (more portable but more complex)
        hdf_path = output_path.with_suffix('.h5')
        df.to_hdf(hdf_path, key='iv_gaussian', mode='w')
        print(f"DataFrame saved to: {hdf_path}")

    else:
        raise ValueError(f"Unknown format: {format}. Use 'pickle' or 'hdf5'")

    # Optionally save scalar columns to CSV
    if save_csv_scalars:
        # Select only scalar columns (exclude arrays)
        scalar_columns = []
        for col in df.columns:
            # Check if column contains arrays
            if not isinstance(df[col].iloc[0], np.ndarray):
                scalar_columns.append(col)

        df_scalars = df[scalar_columns]
        csv_path = output_path.with_suffix('.csv')
        df_scalars.to_csv(csv_path, index=False)
        print(f"Scalar columns saved to: {csv_path}")


def load_dataframe(input_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load DataFrame from file.

    Parameters:
        input_path: Path to saved DataFrame (.pkl or .h5)

    Returns:
        Loaded pandas DataFrame

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format not recognized
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    # Detect format from extension
    if input_path.suffix == '.pkl':
        df = pd.read_pickle(input_path)
        print(f"Loaded DataFrame from: {input_path}")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        return df

    elif input_path.suffix == '.h5':
        df = pd.read_hdf(input_path, key='iv_gaussian')
        print(f"Loaded DataFrame from: {input_path}")
        print(f"  Shape: {df.shape}")
        return df

    else:
        raise ValueError(f"Unrecognized file format: {input_path.suffix}. Use .pkl or .h5")


def get_iv_at_field(df: pd.DataFrame,
                    H_value: float,
                    tolerance: float = 0.01) -> Optional[pd.Series]:
    """
    Get IV data for a specific magnetic field value.

    Parameters:
        df: DataFrame from build_dataframe()
        H_value: Magnetic field value (T)
        tolerance: Tolerance for field matching (T)

    Returns:
        Row from DataFrame closest to H_value, or None if not found
    """
    # Find rows within tolerance
    mask = np.abs(df['H'] - H_value) < tolerance

    if mask.sum() == 0:
        return None

    # Return row closest to target
    idx = (df.loc[mask, 'H'] - H_value).abs().idxmin()
    return df.loc[idx]


def get_current_at_voltage(df: pd.DataFrame,
                          V_value: float,
                          use_filtered: bool = True) -> pd.DataFrame:
    """
    Extract current values at a specific voltage across all H fields.

    Useful for plotting I(H) curves at fixed voltage.

    Parameters:
        df: DataFrame from build_dataframe()
        V_value: Voltage value (V) to extract current at
        use_filtered: Use filtered current (True) or raw current (False)

    Returns:
        DataFrame with columns: H, I_at_V, timestamp
        Sorted by H field
    """
    results = []

    for _, row in df.iterrows():
        # Choose voltage and current arrays
        if use_filtered:
            V = row['voltage_smooth']
            I = row['current_smooth']
        else:
            V = row['voltage']
            I = row['current']

        # Find closest voltage point
        idx = np.argmin(np.abs(V - V_value))
        V_actual = V[idx]
        I_at_V = I[idx]

        results.append({
            'H': row['H'],
            'V_actual': V_actual,
            'I_at_V': I_at_V,
            'timestamp': row['timestamp']
        })

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('timestamp').reset_index(drop=True)

    return result_df


def get_asymmetric_current_at_voltage(df: pd.DataFrame,
                                       V_value: float) -> pd.DataFrame:
    """
    Extract asymmetric current component at a specific voltage across all H fields.

    Useful for analyzing rectification and asymmetric transport effects.

    Parameters:
        df: DataFrame from build_dataframe()
        V_value: Voltage value (V) to extract asymmetric current at

    Returns:
        DataFrame with columns: H, V_actual, I_asym_at_V, timestamp
        Sorted by timestamp to preserve hysteresis order
    """
    results = []

    for _, row in df.iterrows():
        # Use smooth asymmetric component
        V = row['voltage_smooth']
        I_asym = row['current_smooth_asym']

        # Find closest voltage point
        idx = np.argmin(np.abs(V - V_value))
        V_actual = V[idx]
        I_asym_at_V = I_asym[idx]

        results.append({
            'H': row['H'],
            'V_actual': V_actual,
            'I_asym_at_V': I_asym_at_V,
            'timestamp': row['timestamp']
        })

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('timestamp').reset_index(drop=True)

    return result_df
