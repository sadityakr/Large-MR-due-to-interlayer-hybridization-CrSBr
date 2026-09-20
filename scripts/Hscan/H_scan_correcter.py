"""
H_scan Field Correcter

Corrects erroneous zero field values in H_scan data caused by measurement system errors.
The correction is based on detecting non-uniform field steps.

Also calculates signed H values (H_magnitude with sign of Hz component).

Usage:
    from scripts.Hscan.H_scan_correcter import H_scan_correcter, correct_field_data

    # Load data first
    loader = H_scan_dataloader()
    data = loader.load_file('path/to/file.dat')

    # Correct field values
    corrected_data = correct_field_data(data)

    # Access corrected values
    print(corrected_data.Hz_corrected)
    print(corrected_data.H_signed)
"""

from dataclasses import dataclass
import numpy as np
from typing import Tuple

# Handle imports for both module usage and direct execution
try:
    from .H_scan_dataloader import H_scan_data
except ImportError:
    from H_scan_dataloader import H_scan_data


@dataclass
class H_scan_data_corrected(H_scan_data):
    """
    Extended dataclass with corrected field values and signed H.

    Additional attributes beyond H_scan_data:
        Hz_corrected: Corrected Hz field values with erroneous zeros fixed
        H_magnitude_corrected: Recalculated field magnitude with corrected Hz
        H_signed: Signed magnetic field (H_magnitude with sign of Hz)
        num_corrections: Number of field values that were corrected
        corrected_indices: Indices of corrected data points
    """
    Hz_corrected: np.ndarray = None  # type: ignore
    H_magnitude_corrected: np.ndarray = None  # type: ignore
    H_signed: np.ndarray = None  # type: ignore
    num_corrections: int = 0
    corrected_indices: list = None  # type: ignore


def correct_field_data(data: H_scan_data) -> H_scan_data_corrected:
    """
    Correct erroneous zero field values and calculate signed H.

    Algorithm:
    1. Calculate field steps between consecutive points
    2. Identify expected uniform step size
    3. Find points where Hz ≈ 0 and step is non-uniform
    4. Replace erroneous zeros with interpolated values
    5. Calculate signed H = sign(Hz_corrected) * |H|

    Args:
        data: H_scan_data object from H_scan_dataloader

    Returns:
        H_scan_data_corrected object with corrected field values
    """
    # Correct Hz field values
    Hz_corrected, corrected_indices = _correct_Hz_field(data.Hz_T)

    # Recalculate H magnitude with corrected Hz
    H_magnitude_corrected = np.sqrt(data.Hx_T**2 + data.Hy_T**2 + Hz_corrected**2)

    # Calculate signed H (magnitude with sign of Hz)
    H_signed = np.sign(Hz_corrected) * H_magnitude_corrected

    # Create extended dataclass
    corrected_data = H_scan_data_corrected(
        # Copy all original fields
        filename=data.filename,
        date=data.date,
        time=data.time,
        Tcryo_K=data.Tcryo_K,
        Tsample_K=data.Tsample_K,
        Hx_T=data.Hx_T,
        Hy_T=data.Hy_T,
        Hz_T=data.Hz_T,
        H_mag=data.H_mag,
        phi=data.phi,
        theta=data.theta,
        L1_Ch1=data.L1_Ch1,
        L1_Ch2=data.L1_Ch2,
        L2_Ch1=data.L2_Ch1,
        L2_Ch2=data.L2_Ch2,
        L3_Ch1=data.L3_Ch1,
        L3_Ch2=data.L3_Ch2,
        L4_Ch1=data.L4_Ch1,
        L4_Ch2=data.L4_Ch2,
        L5_Ch1=data.L5_Ch1,
        L5_Ch2=data.L5_Ch2,
        # Add corrected fields
        Hz_corrected=Hz_corrected,
        H_magnitude_corrected=H_magnitude_corrected,
        H_signed=H_signed,
        num_corrections=len(corrected_indices),
        corrected_indices=corrected_indices
    )

    return corrected_data


def _correct_Hz_field(Hz_values: np.ndarray, threshold: float = 0.001) -> Tuple[np.ndarray, list]:
    """
    Correct erroneous field values in Hz data by detecting isolated outlier points.

    Uses a conservative approach: only corrects points where BOTH adjacent steps
    are significantly non-uniform, indicating an isolated measurement error.

    Args:
        Hz_values: Array of Hz field values in Tesla
        threshold: Not used, kept for API compatibility

    Returns:
        Tuple of (corrected Hz array, list of corrected indices)
    """
    Hz_corrected = Hz_values.copy()
    corrected_indices = []
    n = len(Hz_values)

    if n < 5:
        # Need at least 5 points for correction
        return Hz_corrected, corrected_indices

    # Calculate steps between consecutive points
    steps = np.diff(Hz_values)
    abs_steps = np.abs(steps)

    # Find expected step size using robust statistics
    # Use middle 50% of step sizes to avoid influence of outliers
    sorted_steps = np.sort(abs_steps)
    q25_idx = len(sorted_steps) // 4
    q75_idx = 3 * len(sorted_steps) // 4
    expected_step = np.median(sorted_steps[q25_idx:q75_idx])

    if expected_step < 1e-6:
        # Steps too small, no meaningful correction possible
        return Hz_corrected, corrected_indices

    # Conservative thresholds: only flag truly problematic steps
    # Too small: less than 20% of expected step
    # Too large: more than 300% of expected step
    too_small = expected_step * 0.2
    too_large = expected_step * 3.0

    # For each interior point, check if it creates two bad steps
    for i in range(1, n - 1):
        step_before = abs_steps[i-1]
        step_after = abs_steps[i]

        # Only correct if BOTH adjacent steps are problematic
        # This ensures we only fix isolated bad points, not systematic issues
        before_bad = (step_before < too_small or step_before > too_large)
        after_bad = (step_after < too_small or step_after > too_large)

        if before_bad and after_bad:
            # This point is likely an isolated measurement error
            # Replace with linear interpolation from adjacent good points
            Hz_corrected[i] = (Hz_values[i-1] + Hz_values[i+1]) / 2.0
            corrected_indices.append(i)

    return Hz_corrected, corrected_indices


def get_correction_report(data: H_scan_data_corrected) -> str:
    """
    Generate a text report of field corrections.

    Args:
        data: H_scan_data_corrected object

    Returns:
        String with correction report
    """
    report = f"Field Correction Report\n"
    report += f"{'='*60}\n"
    report += f"File: {data.filename}\n"
    report += f"Total data points: {len(data.Hz_T)}\n"
    report += f"Number of corrections: {data.num_corrections}\n"

    if data.num_corrections > 0:
        report += f"\nCorrected indices: {data.corrected_indices}\n"
        report += f"\nDetails of corrections:\n"
        report += f"{'Index':<8} {'Original Hz (T)':<18} {'Corrected Hz (T)':<18}\n"
        report += f"{'-'*50}\n"

        for idx in data.corrected_indices:
            report += f"{idx:<8} {data.Hz_T[idx]:>16.6f}  {data.Hz_corrected[idx]:>16.6f}\n"
    else:
        report += f"\nNo corrections needed - field data is clean!\n"

    report += f"\nHz field range:\n"
    report += f"  Original:  {data.Hz_T.min():.4f} to {data.Hz_T.max():.4f} T\n"
    report += f"  Corrected: {data.Hz_corrected.min():.4f} to {data.Hz_corrected.max():.4f} T\n"
    report += f"\nSigned H range: {data.H_signed.min():.4f} to {data.H_signed.max():.4f} T\n"
    report += f"{'='*60}\n"

    return report


if __name__ == "__main__":
    # Example usage
    import sys

    # Handle imports for direct execution
    try:
        from H_scan_dataloader import H_scan_dataloader
    except ImportError:
        # If running as a module, try relative import
        from .H_scan_dataloader import H_scan_dataloader

    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        # Default example file
        filepath = str(__import__("pathlib").Path(__file__).resolve().parents[2] / "data" / "device 2" / "b_scans" / "H_scans" / "20251106153846b_scan_30K_92steps_H0.90T_phi_84.0to84.0_theta_0.0to0.0°_v0.dat")

    print("Loading data...")
    loader = H_scan_dataloader()
    data = loader.load_file(filepath)

    print("Correcting field values...")
    corrected_data = correct_field_data(data)

    # Print report
    print("\n" + get_correction_report(corrected_data))

    # Show field step analysis
    print("\nField step analysis:")
    steps_original = np.diff(data.Hz_T)
    steps_corrected = np.diff(corrected_data.Hz_corrected)

    print(f"Original Hz steps:")
    print(f"  Mean: {np.mean(steps_original):.6f} T")
    print(f"  Std:  {np.std(steps_original):.6f} T")
    print(f"  Min:  {np.min(steps_original):.6f} T")
    print(f"  Max:  {np.max(steps_original):.6f} T")

    print(f"\nCorrected Hz steps:")
    print(f"  Mean: {np.mean(steps_corrected):.6f} T")
    print(f"  Std:  {np.std(steps_corrected):.6f} T")
    print(f"  Min:  {np.min(steps_corrected):.6f} T")
    print(f"  Max:  {np.max(steps_corrected):.6f} T")
