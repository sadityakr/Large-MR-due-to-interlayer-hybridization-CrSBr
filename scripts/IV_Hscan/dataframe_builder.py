"""
DataFrame builder module for IV H-scan analysis results.

This module converts analyzed IV data into pandas DataFrames and provides
utilities for error propagation and data export.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Union, Tuple, Optional

from .IV_exp_curvefit import IVAnalyzedData


def build_dataframe(analyzed_dict: Dict[str, IVAnalyzedData],
                   temperature: Optional[float] = None) -> pd.DataFrame:
    """
    Convert analyzed IV data dictionary to pandas DataFrame.

    This function preserves RAW voltage/current arrays for fit verification
    alongside fit parameters and fitted curves.

    Parameters:
        analyzed_dict: Dictionary mapping timestamp → IVAnalyzedData
        temperature: Optional temperature value (K) to add to all rows

    Returns:
        pandas DataFrame with columns:
            Metadata: timestamp, filename, H, Hx, Hy, Hz, temperature
            Fit params: a_fit, a_error, b_fit, b_error, c_fit, c_error
            Quality: r_squared_1, r_squared_2
            Raw data: voltage (array), current (array)
            Fit data: voltage_fit (array), current_fit (array)
    """
    rows = []

    for timestamp, data in analyzed_dict.items():
        row = {
            # Metadata
            'timestamp': data.timestamp,
            'filename': data.filename,
            'H': data.H,
            'Hx': data.Hx,
            'Hy': data.Hy,
            'Hz': data.Hz,

            # Fit parameters (Level 2 - final fit with offset)
            'a_fit': data.a_fit,
            'a_error': data.a_error,
            'b_fit': data.b_fit,
            'b_error': data.b_error,
            'c_fit': data.c_fit,
            'c_error': data.c_error,

            # Goodness of fit
            'r_squared_1': data.r_squared_1,  # First level fit
            'r_squared_2': data.r_squared_2,  # Second level fit (better)

            # RAW DATA (for verification and debugging)
            'voltage': data.voltage,         # Original measured voltages (500 pts)
            'current': data.current,         # Original measured currents (500 pts)

            # FIT DATA (smooth curves for plotting)
            'voltage_fit': data.voltage_fit,   # Smooth voltage array (500 pts)
            'current_fit': data.current_fit,   # Fitted current from level 2
        }

        # Add temperature if provided
        if temperature is not None:
            row['temperature'] = temperature

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by timestamp to preserve hysteresis order
    df = df.sort_values('timestamp').reset_index(drop=True)

    return df


def calculate_current_with_error(V: Union[float, np.ndarray],
                                 a: float, b: float, c: float,
                                 a_err: float, b_err: float, c_err: float) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Calculate current and propagate error using fit parameters.

    The fit function is: I = a * (V - c) * exp(b * |V - c|)

    Error propagation uses partial derivatives:
    σ_I² = (∂I/∂a)²σ_a² + (∂I/∂b)²σ_b² + (∂I/∂c)²σ_c²

    Parameters:
        V: Voltage value(s) (V) - scalar or array
        a: Fit parameter a (A/V)
        b: Fit parameter b (V⁻¹)
        c: Voltage offset (V)
        a_err: Error in a
        b_err: Error in b
        c_err: Error in c

    Returns:
        Tuple of (I_calculated, I_error):
            - I_calculated: Current value(s) (A)
            - I_error: Error in current (A)
    """
    # Calculate voltage offset
    V_offset = V - c

    # Calculate exponential term
    exp_term = np.exp(b * np.abs(V_offset))

    # Calculate current: I = a * (V - c) * exp(b * |V - c|)
    I = a * V_offset * exp_term

    # Partial derivatives for error propagation
    # ∂I/∂a = (V - c) * exp(b * |V - c|)
    dI_da = V_offset * exp_term

    # ∂I/∂b = a * (V - c) * |V - c| * exp(b * |V - c|)
    dI_db = a * V_offset * np.abs(V_offset) * exp_term

    # ∂I/∂c = -a * exp(b * |V - c|) - a * (V - c) * b * sign(V - c) * exp(b * |V - c|)
    # Simplified: -a * exp(b * |V - c|) * (1 + b * |V - c|)
    dI_dc = -a * exp_term * (1 + b * np.abs(V_offset))

    # Propagate error: σ_I² = (∂I/∂a)²σ_a² + (∂I/∂b)²σ_b² + (∂I/∂c)²σ_c²
    I_error_squared = (dI_da * a_err)**2 + (dI_db * b_err)**2 + (dI_dc * c_err)**2
    I_error = np.sqrt(I_error_squared)

    return I, I_error


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
        df.to_hdf(hdf_path, key='iv_analysis', mode='w')
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
        df = pd.read_hdf(input_path, key='iv_analysis')
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
