"""
MR Analysis Module

This module provides functions for magnetoresistance (MR) data analysis,
specifically for extracting data at constant voltage and calculating TMR ratios.

Functions:
----------
- get_data_at_voltage: Extract I_par and I_apar at specified voltage
- calculate_tmr_at_voltage: Calculate TMR = (I_par/I_apar) - 1 for both axes
- propagate_tmr_error: Calculate error propagation for TMR ratio
"""

import numpy as np
import pandas as pd


def get_data_at_voltage(df_combined, voltage, scan_type, tolerance=0.01):
    """
    Extract I_par and I_apar at specified voltage for given scan type.

    Parameters:
    -----------
    df_combined : pd.DataFrame
        Combined dataframe with columns: 'Voltage (V)', 'I_par (A)', 'I_apar (A)',
        'I_par_error (A)', 'I_apar_error (A)', 'temperature', 'scan_type'
    voltage : float
        Target voltage in volts
    scan_type : str
        'b_scan' or 'c_scan'
    tolerance : float, optional
        Voltage tolerance for matching (default: 0.01 V)

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: temperature, I_par, I_par_error,
        I_apar, I_apar_error, voltage_actual
        Sorted by temperature
    """
    # Filter by scan type
    df_scan = df_combined[df_combined['scan_type'] == scan_type]

    # Function to extract data for each temperature group
    def extract_at_voltage(group):
        voltage_diff = np.abs(group['Voltage (V)'] - voltage)
        closest_idx = voltage_diff.idxmin()

        if voltage_diff.loc[closest_idx] < tolerance:
            return pd.Series({
                'I_par': group.loc[closest_idx, 'I_par (A)'],
                'I_par_error': group.loc[closest_idx, 'I_par_error (A)'],
                'I_apar': group.loc[closest_idx, 'I_apar (A)'],
                'I_apar_error': group.loc[closest_idx, 'I_apar_error (A)'],
                'voltage_actual': group.loc[closest_idx, 'Voltage (V)']
            })
        else:
            return pd.Series({
                'I_par': np.nan,
                'I_par_error': np.nan,
                'I_apar': np.nan,
                'I_apar_error': np.nan,
                'voltage_actual': np.nan
            })

    # Apply to each temperature group
    result = df_scan.groupby('temperature').apply(extract_at_voltage).reset_index()

    return result.sort_values('temperature')


def propagate_tmr_error(I_par, I_par_error, I_apar, I_apar_error):
    """
    Calculate error for TMR = (I_par/I_apar) - 1

    Error propagation formula:
    error(TMR) = |I_par/I_apar| * sqrt((I_par_error/I_par)^2 + (I_apar_error/I_apar)^2)

    Since d/dx(x-1) = 1, the error in (ratio - 1) equals the error in ratio.

    Parameters:
    -----------
    I_par : float or array
        Parallel current
    I_par_error : float or array
        Error in parallel current
    I_apar : float or array
        Anti-parallel current
    I_apar_error : float or array
        Error in anti-parallel current

    Returns:
    --------
    float or array
        Error in TMR ratio
    """
    ratio = I_par / I_apar
    ratio_error = np.abs(ratio) * np.sqrt(
        (I_par_error / I_par)**2 + (I_apar_error / I_apar)**2
    )
    return ratio_error


def calculate_tmr_at_voltage(df_combined, voltage, tolerance=0.01):
    """
    Calculate TMR ratio at specified voltage for both b and c axes.

    TMR = (I_par / I_apar) - 1

    Parameters:
    -----------
    df_combined : pd.DataFrame
        Combined dataframe with all scan data
    voltage : float
        Target voltage in volts
    tolerance : float, optional
        Voltage tolerance for matching (default: 0.01 V)

    Returns:
    --------
    dict
        Dictionary with keys 'b_axis' and 'c_axis', each containing a DataFrame with:
        - temperature
        - TMR (TMR ratio)
        - TMR_error (error in TMR)
        - I_par, I_par_error
        - I_apar, I_apar_error
        - voltage_actual
    """
    result = {}

    for scan_type, axis_name in [('b_scan', 'b_axis'), ('c_scan', 'c_axis')]:
        # Extract data at voltage
        data = get_data_at_voltage(df_combined, voltage, scan_type, tolerance)

        # Calculate TMR = (I_par / I_apar) - 1
        data['TMR'] = (data['I_par'] / data['I_apar']) - 1

        # Calculate TMR error
        data['TMR_error'] = propagate_tmr_error(
            data['I_par'], data['I_par_error'],
            data['I_apar'], data['I_apar_error']
        )

        result[axis_name] = data

    return result
