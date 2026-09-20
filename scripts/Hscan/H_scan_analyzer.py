"""
H_scan Data Analyzer

Analyzes H_scan data by applying field corrections and calculating channel statistics.
Computes min/max values for lock-in channels using robust statistics.

Usage:
    from scripts.Hscan.H_scan_analyzer import analyze_H_scan_data

    # Load data first
    loader = H_scan_dataloader()
    raw_data = loader.load_file('path/to/file.dat')

    # Analyze
    analyzed_data = analyze_H_scan_data(raw_data)

    # Access results
    print(analyzed_data.L1_min, analyzed_data.L1_min_error)
    print(analyzed_data.H_signed)
"""

from dataclasses import dataclass
import numpy as np
from typing import Tuple

# Handle imports for both module usage and direct execution
try:
    from .H_scan_dataloader import H_scan_data
    from .H_scan_correcter import correct_field_data, H_scan_data_corrected
except ImportError:
    from H_scan_dataloader import H_scan_data
    from H_scan_correcter import correct_field_data, H_scan_data_corrected


@dataclass
class H_scan_data_analyzed(H_scan_data_corrected):
    """
    Extended dataclass with field corrections and channel statistics.

    Additional attributes beyond H_scan_data_corrected:
        L1_min: Minimum value of L1_Ch1 (mean of 10 lowest values)
        L1_max: Maximum value of L1_Ch1 (mean of 10 highest values)
        L1_min_error: Standard error of L1_Ch1 minimum
        L1_max_error: Standard error of L1_Ch1 maximum

        L2_min, L2_max, L2_min_error, L2_max_error: Same for L2_Ch1
        L3_min, L3_max, L3_min_error, L3_max_error: Same for L3_Ch1
        L4_min, L4_max, L4_min_error, L4_max_error: Same for L4_Ch1
    """
    # L1 statistics
    L1_min: float = None  # type: ignore
    L1_max: float = None  # type: ignore
    L1_min_error: float = None  # type: ignore
    L1_max_error: float = None  # type: ignore

    # L2 statistics
    L2_min: float = None  # type: ignore
    L2_max: float = None  # type: ignore
    L2_min_error: float = None  # type: ignore
    L2_max_error: float = None  # type: ignore

    # L3 statistics
    L3_min: float = None  # type: ignore
    L3_max: float = None  # type: ignore
    L3_min_error: float = None  # type: ignore
    L3_max_error: float = None  # type: ignore

    # L4 statistics
    L4_min: float = None  # type: ignore
    L4_max: float = None  # type: ignore
    L4_min_error: float = None  # type: ignore
    L4_max_error: float = None  # type: ignore


def analyze_H_scan_data(data: H_scan_data, n_points: int = 10) -> H_scan_data_analyzed:
    """
    Analyze H_scan data: correct fields and calculate channel statistics.

    Args:
        data: H_scan_data object from H_scan_dataloader
        n_points: Number of extreme points to use for min/max calculation (default: 10)

    Returns:
        H_scan_data_analyzed object with corrections and statistics
    """
    # Step 1: Apply field corrections
    corrected_data = correct_field_data(data)

    # Step 2: Calculate statistics for each lock-in channel
    L1_stats = _calculate_channel_stats(data.L1_Ch1, n_points)
    L2_stats = _calculate_channel_stats(data.L2_Ch1, n_points)
    L3_stats = _calculate_channel_stats(data.L3_Ch1, n_points)
    L4_stats = _calculate_channel_stats(data.L4_Ch1, n_points)

    # Step 3: Create analyzed dataclass
    analyzed_data = H_scan_data_analyzed(
        # Copy all fields from corrected data
        filename=corrected_data.filename,
        date=corrected_data.date,
        time=corrected_data.time,
        Tcryo_K=corrected_data.Tcryo_K,
        Tsample_K=corrected_data.Tsample_K,
        Hx_T=corrected_data.Hx_T,
        Hy_T=corrected_data.Hy_T,
        Hz_T=corrected_data.Hz_T,
        H_mag=corrected_data.H_mag,
        phi=corrected_data.phi,
        theta=corrected_data.theta,
        L1_Ch1=corrected_data.L1_Ch1,
        L1_Ch2=corrected_data.L1_Ch2,
        L2_Ch1=corrected_data.L2_Ch1,
        L2_Ch2=corrected_data.L2_Ch2,
        L3_Ch1=corrected_data.L3_Ch1,
        L3_Ch2=corrected_data.L3_Ch2,
        L4_Ch1=corrected_data.L4_Ch1,
        L4_Ch2=corrected_data.L4_Ch2,
        L5_Ch1=corrected_data.L5_Ch1,
        L5_Ch2=corrected_data.L5_Ch2,
        # Corrected fields
        Hz_corrected=corrected_data.Hz_corrected,
        H_magnitude_corrected=corrected_data.H_magnitude_corrected,
        H_signed=corrected_data.H_signed,
        num_corrections=corrected_data.num_corrections,
        corrected_indices=corrected_data.corrected_indices,
        # Channel statistics
        L1_min=L1_stats['min'],
        L1_max=L1_stats['max'],
        L1_min_error=L1_stats['min_error'],
        L1_max_error=L1_stats['max_error'],
        L2_min=L2_stats['min'],
        L2_max=L2_stats['max'],
        L2_min_error=L2_stats['min_error'],
        L2_max_error=L2_stats['max_error'],
        L3_min=L3_stats['min'],
        L3_max=L3_stats['max'],
        L3_min_error=L3_stats['min_error'],
        L3_max_error=L3_stats['max_error'],
        L4_min=L4_stats['min'],
        L4_max=L4_stats['max'],
        L4_min_error=L4_stats['min_error'],
        L4_max_error=L4_stats['max_error'],
    )

    return analyzed_data


def _calculate_channel_stats(channel_data: np.ndarray, n_points: int = 10) -> dict:
    """
    Calculate min/max statistics for a lock-in channel.

    Uses mean of n extreme values and standard error for robust estimation.

    Args:
        channel_data: Array of channel values
        n_points: Number of extreme points to average (default: 10)

    Returns:
        Dictionary with keys: 'min', 'max', 'min_error', 'max_error'
    """
    # Remove any NaN values
    clean_data = channel_data[~np.isnan(channel_data)]

    if len(clean_data) < n_points:
        # Not enough data points
        return {
            'min': np.nan,
            'max': np.nan,
            'min_error': np.nan,
            'max_error': np.nan
        }

    # Sort data
    sorted_data = np.sort(np.abs(clean_data))

    # Get n lowest and n highest values
    lowest_n = sorted_data[:n_points]
    highest_n = sorted_data[-n_points:]

    # Calculate means
    min_value = np.mean(lowest_n)
    max_value = np.mean(highest_n)

    # Calculate standard errors
    # Standard error = std / sqrt(n)
    min_error = np.std(lowest_n, ddof=1) / np.sqrt(n_points)
    max_error = np.std(highest_n, ddof=1) / np.sqrt(n_points)

    return {
        'min': min_value,
        'max': max_value,
        'min_error': min_error,
        'max_error': max_error
    }


def get_analysis_summary(data: H_scan_data_analyzed) -> str:
    """
    Generate a text summary of the analysis results.

    Args:
        data: H_scan_data_analyzed object

    Returns:
        String with analysis summary
    """
    summary = f"H_scan Data Analysis Summary\n"
    summary += f"{'='*70}\n"
    summary += f"File: {data.filename}\n"
    summary += f"Data points: {len(data.Hz_T)}\n"
    summary += f"Temperature: {data.Tcryo_K[0]:.1f} K\n"
    summary += f"\nField Corrections:\n"
    summary += f"  Number of corrections: {data.num_corrections}\n"
    summary += f"  Hz range (corrected): {data.Hz_corrected.min():.4f} to {data.Hz_corrected.max():.4f} T\n"
    summary += f"  H_signed range: {data.H_signed.min():.4f} to {data.H_signed.max():.4f} T\n"

    summary += f"\nChannel Statistics (mean of 10 extreme values):\n"
    summary += f"{'-'*70}\n"
    summary += f"{'Channel':<12} {'Min Value':<15} {'Min Error':<15} {'Max Value':<15} {'Max Error':<15}\n"
    summary += f"{'-'*70}\n"

    channels = [
        ('L1_Ch1', data.L1_min, data.L1_min_error, data.L1_max, data.L1_max_error),
        ('L2_Ch1', data.L2_min, data.L2_min_error, data.L2_max, data.L2_max_error),
        ('L3_Ch1', data.L3_min, data.L3_min_error, data.L3_max, data.L3_max_error),
        ('L4_Ch1', data.L4_min, data.L4_min_error, data.L4_max, data.L4_max_error),
    ]

    for ch_name, min_val, min_err, max_val, max_err in channels:
        summary += f"{ch_name:<12} {min_val:<15.6e} {min_err:<15.6e} {max_val:<15.6e} {max_err:<15.6e}\n"

    summary += f"{'='*70}\n"

    return summary


if __name__ == "__main__":
    # Example usage
    import sys

    # Handle imports for direct execution
    try:
        from H_scan_dataloader import H_scan_dataloader
    except ImportError:
        from .H_scan_dataloader import H_scan_dataloader

    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        # Default example file
        filepath = str(__import__("pathlib").Path(__file__).resolve().parents[2] / "data" / "device 2" / "b_scans" / "H_scans" / "20251106153846b_scan_30K_92steps_H0.90T_phi_84.0to84.0_theta_0.0to0.0°_v0.dat")

    print("Loading data...")
    loader = H_scan_dataloader()
    raw_data = loader.load_file(filepath)

    print("Analyzing data...")
    analyzed_data = analyze_H_scan_data(raw_data)

    # Print summary
    print("\n" + get_analysis_summary(analyzed_data))

    # Additional details
    print("\nChannel Statistics Details:")
    print(f"L1: Min = {analyzed_data.L1_min:.6e} ± {analyzed_data.L1_min_error:.6e}")
    print(f"    Max = {analyzed_data.L1_max:.6e} ± {analyzed_data.L1_max_error:.6e}")
    print(f"    Range = {analyzed_data.L1_max - analyzed_data.L1_min:.6e}")
