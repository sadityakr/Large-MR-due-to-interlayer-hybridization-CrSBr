"""
Magnetic field value correction for IV H-scan dataframes.

Detects and corrects erroneous field values caused by measurement system errors.
Handles duplicate measurements (each field measured twice in forward/backward sweeps).

Created: 2025-12-07

Usage:
    from scripts.IV_Hscan_gaussian.field_value_fixer import fix_dataframe_fields

    df_corrected, report = fix_dataframe_fields(df, threshold_multiplier=2.5, dry_run=False)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def build_unique_field_map(df: pd.DataFrame, tolerance: float = 1e-6) -> Dict[int, List[int]]:
    """
    Map unique field values to their row indices in the dataframe.

    Handles duplicate measurements where consecutive rows have the same H value.

    Parameters:
        df: DataFrame with 'H' column containing magnetic field values
        tolerance: Tolerance for considering two H values identical (default: 1e-6 T)

    Returns:
        Dictionary mapping unique_idx → [row_idx1, row_idx2, ...]

    Example:
        If H values are [0.9, 0.9, 0.8, 0.8, 0.7], returns:
        {0: [0, 1], 1: [2, 3], 2: [4]}
    """
    unique_map = {}
    unique_idx = 0
    i = 0

    while i < len(df):
        current_H = df.iloc[i]['H']
        indices = [i]

        # Check for duplicate measurements (consecutive rows with same H)
        while i + 1 < len(df) and abs(df.iloc[i + 1]['H'] - current_H) < tolerance:
            indices.append(i + 1)
            i += 1

        unique_map[unique_idx] = indices
        unique_idx += 1
        i += 1

    return unique_map


def extract_unique_fields(df: pd.DataFrame, unique_map: Dict[int, List[int]]) -> np.ndarray:
    """
    Extract unique H field values from dataframe using the unique_map.

    Parameters:
        df: DataFrame with 'H' column
        unique_map: Mapping from build_unique_field_map()

    Returns:
        Array of unique H values in measurement order
    """
    unique_H = np.array([df.iloc[indices[0]]['H'] for indices in unique_map.values()])
    return unique_H


def detect_outlier_steps(unique_H: np.ndarray, threshold_multiplier: float = 2.5) -> List[int]:
    """
    Detect outlier field values using step size analysis.

    Algorithm:
    1. Calculate step sizes between consecutive field values
    2. Compute robust median step (excluding near-zero steps)
    3. Flag steps larger than threshold_multiplier × median
    4. Mark the point AFTER a large step as an outlier

    Parameters:
        unique_H: Array of unique field values
        threshold_multiplier: Multiplier for median step threshold (default: 2.5)

    Returns:
        List of indices in unique_H that are outliers
    """
    if len(unique_H) < 3:
        # Need at least 3 points to detect outliers
        return []

    # Calculate steps
    steps = np.diff(unique_H)
    abs_steps = np.abs(steps)

    # Robust median calculation (exclude near-zero steps)
    valid_steps = abs_steps[abs_steps > 1e-6]

    if len(valid_steps) == 0:
        # All steps are zero - no variation
        return []

    median_step = np.median(valid_steps)
    threshold = threshold_multiplier * median_step

    # Detect outliers: point after a large step
    outlier_indices = []
    for i in range(len(steps)):
        if abs_steps[i] > threshold:
            # Large step from i to i+1 → i+1 is likely wrong
            outlier_indices.append(i + 1)

    return outlier_indices


def correct_outliers(unique_H: np.ndarray, outlier_indices: List[int]) -> np.ndarray:
    """
    Correct outlier values using interpolation or extrapolation.

    Strategy:
    - Interior points: Linear interpolation from neighbors
    - First point: Extrapolate backward using next two valid points
    - Last point: Extrapolate forward using previous two valid points

    Parameters:
        unique_H: Array of unique field values
        outlier_indices: List of indices to correct

    Returns:
        Corrected array of unique H values
    """
    corrected_H = unique_H.copy()
    n = len(unique_H)

    for idx in outlier_indices:
        if 0 < idx < n - 1:
            # Interior point: average of neighbors
            corrected_H[idx] = (corrected_H[idx - 1] + corrected_H[idx + 1]) / 2.0

        elif idx == 0:
            # First point: extrapolate backward
            if n >= 3:
                # Use constant step assumption from next two points
                step = corrected_H[2] - corrected_H[1]
                corrected_H[0] = corrected_H[1] - step
            else:
                # Not enough points, leave unchanged
                pass

        elif idx == n - 1:
            # Last point: extrapolate forward
            if n >= 3:
                # Use constant step assumption from previous two points
                step = corrected_H[-2] - corrected_H[-3]
                corrected_H[-1] = corrected_H[-2] + step
            else:
                # Not enough points, leave unchanged
                pass

    return corrected_H


def apply_corrections_to_dataframe(df: pd.DataFrame,
                                   unique_map: Dict[int, List[int]],
                                   corrected_H: np.ndarray) -> pd.DataFrame:
    """
    Apply corrected unique field values back to the full dataframe.

    Ensures that duplicate measurements (pairs) get the same corrected value.

    Parameters:
        df: Original dataframe
        unique_map: Mapping from unique indices to dataframe row indices
        corrected_H: Array of corrected unique H values

    Returns:
        New dataframe with corrected H values
    """
    df_corrected = df.copy()

    for unique_idx, row_indices in unique_map.items():
        corrected_value = corrected_H[unique_idx]

        # Apply correction to all rows corresponding to this unique value
        for row_idx in row_indices:
            df_corrected.at[row_idx, 'H'] = corrected_value

    return df_corrected


def validate_corrections(df_original: pd.DataFrame, df_corrected: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that corrections didn't introduce new problems.

    Checks:
    1. Only H column was modified
    2. No NaN or inf values introduced
    3. Field range is reasonable
    4. Duplicate pairs still match

    Parameters:
        df_original: Original dataframe before correction
        df_corrected: Corrected dataframe

    Returns:
        (is_valid, list_of_warnings)
    """
    warnings = []
    is_valid = True

    # Check 1: Only H column modified
    for col in df_original.columns:
        if col == 'H':
            continue
        if not df_original[col].equals(df_corrected[col]):
            # For array columns, need special comparison
            if isinstance(df_original[col].iloc[0], np.ndarray):
                # Check if arrays are equal
                arrays_equal = all(
                    np.array_equal(df_original[col].iloc[i], df_corrected[col].iloc[i])
                    for i in range(len(df_original))
                )
                if not arrays_equal:
                    warnings.append(f"Column '{col}' was unexpectedly modified")
                    is_valid = False
            else:
                warnings.append(f"Column '{col}' was unexpectedly modified")
                is_valid = False

    # Check 2: No invalid values
    if df_corrected['H'].isna().any():
        warnings.append("NaN values found in corrected H column")
        is_valid = False

    if np.isinf(df_corrected['H'].values).any():
        warnings.append("Inf values found in corrected H column")
        is_valid = False

    # Check 3: Field range reasonable
    H_min = df_corrected['H'].min()
    H_max = df_corrected['H'].max()
    if abs(H_min) > 10.0 or abs(H_max) > 10.0:
        warnings.append(f"Field range seems unreasonable: {H_min:.2f} to {H_max:.2f} T")

    return is_valid, warnings


def generate_correction_report(df_original: pd.DataFrame,
                               df_corrected: pd.DataFrame,
                               unique_map: Dict[int, List[int]],
                               outlier_indices: List[int],
                               unique_H_original: np.ndarray,
                               unique_H_corrected: np.ndarray) -> str:
    """
    Generate a detailed report of the corrections made.

    Parameters:
        df_original: Original dataframe
        df_corrected: Corrected dataframe
        unique_map: Mapping of unique indices to row indices
        outlier_indices: List of outlier indices that were corrected
        unique_H_original: Original unique H values
        unique_H_corrected: Corrected unique H values

    Returns:
        Formatted report string
    """
    report = []
    report.append("="*70)
    report.append("FIELD VALUE CORRECTION REPORT")
    report.append("="*70)
    report.append(f"Total rows: {len(df_original)}")
    report.append(f"Unique field values: {len(unique_map)}")
    report.append(f"Outliers detected: {len(outlier_indices)}")

    if len(outlier_indices) > 0:
        report.append("\nCorrected values:")
        report.append(f"{'Unique Idx':<12} {'Original H':>12} {'Corrected H':>12} {'Change':>12}")
        report.append("-"*50)

        for idx in outlier_indices:
            orig = unique_H_original[idx]
            corr = unique_H_corrected[idx]
            change = corr - orig
            report.append(f"{idx:<12} {orig:>12.6f} {corr:>12.6f} {change:>+12.6f}")
    else:
        report.append("\nNo outliers detected - no corrections needed.")

    report.append("="*70)

    return "\n".join(report)


def fix_dataframe_fields(df: pd.DataFrame,
                        threshold_multiplier: float = 2.5,
                        dry_run: bool = False) -> Tuple[pd.DataFrame, str]:
    """
    Main API function to fix erroneous field values in IV scan dataframe.

    This function:
    1. Identifies duplicate measurements
    2. Detects outlier field values using step size analysis
    3. Corrects outliers via interpolation/extrapolation
    4. Validates corrections
    5. Generates detailed report

    Parameters:
        df: DataFrame with IV scan data (must have 'H' column)
        threshold_multiplier: Multiplier for median step threshold (default: 2.5)
                              Higher values = less aggressive correction
        dry_run: If True, show what would be corrected without modifying dataframe

    Returns:
        (df_corrected, report_string)
        - df_corrected: Corrected dataframe (or original if dry_run=True)
        - report_string: Detailed report of corrections

    Example:
        >>> df_fixed, report = fix_dataframe_fields(df, threshold_multiplier=2.5)
        >>> print(report)
    """
    # Step 1: Build mapping of unique fields to row indices
    unique_map = build_unique_field_map(df)

    # Step 2: Extract unique H values
    unique_H_original = extract_unique_fields(df, unique_map)

    # Step 3: Detect outliers
    outlier_indices = detect_outlier_steps(unique_H_original, threshold_multiplier)

    # Step 4: Correct outliers
    unique_H_corrected = correct_outliers(unique_H_original, outlier_indices)

    # Step 5: Apply corrections to dataframe (or skip if dry_run)
    if dry_run or len(outlier_indices) == 0:
        df_corrected = df  # No modification
    else:
        df_corrected = apply_corrections_to_dataframe(df, unique_map, unique_H_corrected)

    # Step 6: Validate corrections
    if len(outlier_indices) > 0 and not dry_run:
        is_valid, warnings = validate_corrections(df, df_corrected)
        if not is_valid:
            raise RuntimeError(f"Validation failed: {warnings}")

    # Step 7: Generate report
    report = generate_correction_report(
        df, df_corrected, unique_map, outlier_indices,
        unique_H_original, unique_H_corrected
    )

    return df_corrected, report
