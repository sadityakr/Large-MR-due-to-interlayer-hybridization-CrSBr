"""
IV Gaussian Filter Analysis
Created: 09/11/2025

Applies Gaussian filtering to IV data and estimates voltage offset from zero-crossing.
Extends IVDataExtended with filtered results and quality metrics.
"""
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from typing import Optional, Tuple
from ..IV_Hscan.IV_H_scan_data_loader import IVDataExtended, load_iv_data, data_parser_IV


@dataclass
class IVFilteredData:
    """
    Extended data container with Gaussian filter results.

    Attributes:
        filename: Name of the file (without path)
        filepath: Full path to the data file
        voltage: Array of voltage values (V)
        current: Array of current values (A)
        Hx: Magnetic field x-component (T)
        Hy: Magnetic field y-component (T)
        Hz: Magnetic field z-component (T)
        H: Total magnetic field magnitude with sign (T)
        timestamp: Measurement timestamp

        # Outlier removal
        outlier_mask: Boolean array marking outlier points (True = outlier)
        n_outliers_removed: Number of outliers removed
        voltage_clean: Voltage array with outliers removed
        current_clean: Current array with outliers removed

        # Filter parameters
        sigma: Gaussian filter sigma value
        truncate: Truncate filter at this many standard deviations

        # Filtered results
        current_filtered: Gaussian filtered current values
        v_offset: Voltage offset where filtered current crosses zero (V)
        residual_rms: RMS residual between raw and filtered current (A)

        # Smooth arrays for plotting
        voltage_smooth: Smooth voltage array (500 pts)
        current_smooth: Smooth filtered current (500 pts)

        # Symmetric/asymmetric decomposition
        current_filtered_sym: Symmetric component of filtered current
        current_filtered_asym: Asymmetric component of filtered current
        current_smooth_sym: Smooth symmetric component (500 pts)
        current_smooth_asym: Smooth asymmetric component (500 pts)
    """
    # Base data from IVDataExtended
    filename: str
    filepath: str
    voltage: np.ndarray
    current: np.ndarray
    Hx: float
    Hy: float
    Hz: float
    H: float
    timestamp: str

    # Outlier removal
    outlier_mask: np.ndarray
    n_outliers_removed: int
    voltage_clean: np.ndarray
    current_clean: np.ndarray

    # Filter parameters
    sigma: float
    truncate: float

    # Filtered results
    current_filtered: np.ndarray
    v_offset: float
    residual_rms: float

    # Smooth arrays for plotting
    voltage_smooth: np.ndarray
    current_smooth: np.ndarray

    # Symmetric/asymmetric decomposition
    current_filtered_sym: np.ndarray
    current_filtered_asym: np.ndarray
    current_smooth_sym: np.ndarray
    current_smooth_asym: np.ndarray


def detect_outliers(voltage: np.ndarray,
                   current: np.ndarray,
                   mad_threshold: float = 30.0,
                   current_threshold_percentile: float = 10.0) -> np.ndarray:
    """
    Simple MAD-based outlier detection in log space.

    For exponential IV curves, taking log(abs(I)) linearizes the relationship
    with voltage, making MAD filtering much more effective and simple.

    Algorithm:
    1. Exclude very low current points (near zero-crossing) from analysis
    2. Convert remaining current to log space: log(abs(I))
    3. Apply Gaussian smoothing to create reference curve
    4. Calculate residuals from reference
    5. Use MAD to identify outliers that deviate significantly

    Parameters:
        voltage: Voltage array (V)
        current: Current array (A)
        mad_threshold: Threshold in MAD units for outlier detection (default: 30.0)
                       Higher values = less aggressive filtering. Typical range: 2-5.
        current_threshold_percentile: Exclude bottom percentile of current values (default: 10.0)

    Returns:
        Boolean array where True indicates an outlier
    """
    from scipy.ndimage import gaussian_filter1d

    # Sort by voltage
    sorted_indices = np.argsort(voltage)
    V_sorted = voltage[sorted_indices]
    I_sorted = current[sorted_indices]

    # Find threshold to exclude very low currents (near zero crossing)
    I_abs = np.abs(I_sorted)
    current_threshold = np.percentile(I_abs, current_threshold_percentile)

    # Mask for points with sufficient current (not near zero)
    high_current_mask = I_abs > current_threshold

    # Replace zeros with minimum positive value
    min_positive = np.min(I_abs[I_abs > 0]) if np.any(I_abs > 0) else 1e-12
    I_abs[I_abs == 0] = min_positive

    log_I = np.log(I_abs)

    # Apply Gaussian filter to log(I) - use moderate sigma
    log_I_smooth = gaussian_filter1d(log_I, sigma=5.0, truncate=4.0)

    # Calculate residuals in log space ONLY for high current points
    residuals = log_I - log_I_smooth
    residuals_high_current = residuals[high_current_mask]

    # Use MAD for robust outlier detection (only on high current region)
    residual_median = np.median(residuals_high_current)
    residual_mad = np.median(np.abs(residuals_high_current - residual_median))

    # Convert MAD to standard deviation: σ ≈ 1.4826 * MAD
    if residual_mad > 1e-12:
        residual_std = 1.4826 * residual_mad
        threshold_value = mad_threshold * residual_std
    else:
        # Fallback to std if MAD is too small
        residual_std = np.std(residuals_high_current)
        threshold_value = mad_threshold * residual_std

    # Identify outliers (only check high current points)
    outliers_sorted = np.zeros(len(voltage), dtype=bool)
    outliers_sorted[high_current_mask] = np.abs(residuals[high_current_mask] - residual_median) > threshold_value

    # Map back to original indices
    outliers = np.zeros(len(voltage), dtype=bool)
    for idx in np.where(outliers_sorted)[0]:
        orig_idx = sorted_indices[idx]
        outliers[orig_idx] = True

    return outliers




def remove_outliers(voltage: np.ndarray,
                   current: np.ndarray,
                   outlier_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove outlier points from voltage and current arrays.

    Parameters:
        voltage: Voltage array (V)
        current: Current array (A)
        outlier_mask: Boolean array (True = outlier)

    Returns:
        Tuple of (voltage_clean, current_clean) with outliers removed
    """
    clean_mask = ~outlier_mask
    return voltage[clean_mask], current[clean_mask]


def decompose_symmetric_asymmetric(voltage: np.ndarray,
                                   current: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose current into symmetric and asymmetric components along voltage axis.

    Symmetric component:  I_sym(V) = [I(V) + I(-V)] / 2  (even function)
    Asymmetric component: I_asym(V) = [I(V) - I(-V)] / 2  (odd function)

    Parameters:
        voltage: Voltage array (V)
        current: Current array (A)

    Returns:
        Tuple of (I_sym, I_asym) arrays with same shape as input

    Notes:
        - I_sym(-V) = I_sym(V)
        - I_asym(-V) = -I_asym(V)
        - I(V) = I_sym(V) + I_asym(V)
    """
    from scipy.interpolate import interp1d

    # Remove duplicate voltage values by averaging current at duplicate points
    V_unique, unique_indices = np.unique(voltage, return_inverse=True)
    I_unique = np.array([current[unique_indices == i].mean()
                        for i in range(len(V_unique))])

    # Create interpolator for I(V) using deduplicated data
    # Use linear interpolation for smooth results
    I_interp = interp1d(V_unique, I_unique, kind='cubic',
                        bounds_error=False, fill_value='extrapolate')

    # Calculate I(-V) for each voltage point (using original voltage array)
    I_at_minus_V = I_interp(-voltage)

    # Calculate symmetric and asymmetric components
    I_sym = (current + I_at_minus_V) / 2.0
    I_asym = (current - I_at_minus_V) / 2.0

    return I_sym, I_asym


def find_voltage_offset(voltage: np.ndarray, current_filtered: np.ndarray) -> float:
    """
    Find voltage offset where filtered current crosses zero.

    Uses linear interpolation to find the zero-crossing point.
    If multiple crossings exist, returns the one closest to V=0.

    Parameters:
        voltage: Voltage array (V)
        current_filtered: Filtered current array (A)

    Returns:
        Voltage offset (V) where I_filtered = 0
        Returns NaN if no zero-crossing found
    """
    # Check if current crosses zero
    sign_changes = np.diff(np.sign(current_filtered))
    zero_crossings = np.where(sign_changes != 0)[0]

    if len(zero_crossings) == 0:
        # No zero crossing found
        return np.nan

    # Find all zero-crossing points using linear interpolation
    v_offset_candidates = []
    for idx in zero_crossings:
        # Linear interpolation between idx and idx+1
        v1, v2 = voltage[idx], voltage[idx + 1]
        i1, i2 = current_filtered[idx], current_filtered[idx + 1]

        # V_offset = v1 - i1 * (v2 - v1) / (i2 - i1)
        if i2 != i1:  # Avoid division by zero
            v_offset = v1 - i1 * (v2 - v1) / (i2 - i1)
            v_offset_candidates.append(v_offset)

    if len(v_offset_candidates) == 0:
        return np.nan

    # Return crossing closest to V=0
    v_offset_candidates = np.array(v_offset_candidates)
    closest_idx = np.argmin(np.abs(v_offset_candidates))
    return v_offset_candidates[closest_idx]


def calculate_residual_rms(current_raw: np.ndarray,
                           current_filtered: np.ndarray) -> float:
    """
    Calculate RMS residual between raw and filtered current.

    Parameters:
        current_raw: Raw current measurements (A)
        current_filtered: Filtered current values (A)

    Returns:
        RMS residual (A)
    """
    residuals = current_raw - current_filtered
    return np.sqrt(np.mean(residuals**2))


def filter_iv_data(extended_data: IVDataExtended,
                   sigma: float = 1.5,
                   truncate: float = 4.0,
                   remove_outliers_flag: bool = True,
                   mad_threshold: float = 30.0) -> IVFilteredData:
    """
    Apply Gaussian filter to IV data and calculate voltage offset.

    This function optionally removes outliers before filtering using a simple
    MAD-based approach in log space, which is ideal for exponential IV curves.

    Parameters:
        extended_data: IVDataExtended object with IV data
        sigma: Standard deviation for Gaussian kernel for final filtering (default: 1.5)
        truncate: Truncate filter at this many standard deviations (default: 4.0)
        remove_outliers_flag: Whether to remove outliers before filtering (default: True)
        mad_threshold: Threshold in MAD units for outlier detection in log space (default: 30.0)
                       Higher values = less aggressive filtering. Typical range: 2-5.

    Returns:
        IVFilteredData object with filtered results and quality metrics

    Raises:
        RuntimeError: If filtering fails
    """
    try:
        V = extended_data.voltage
        I = extended_data.current

        # Step 1: Detect and remove outliers (if enabled)
        if remove_outliers_flag:
            outlier_mask = detect_outliers(V, I, mad_threshold=mad_threshold)
            V_clean, I_clean = remove_outliers(V, I, outlier_mask)
            n_outliers = outlier_mask.sum()
        else:
            outlier_mask = np.zeros(len(V), dtype=bool)
            V_clean, I_clean = V, I
            n_outliers = 0

        # Step 2: Apply Gaussian filter to cleaned current
        I_filtered = gaussian_filter1d(I_clean, sigma=sigma, truncate=truncate)

        # Step 3: Find voltage offset from zero-crossing (using clean data)
        v_offset = find_voltage_offset(V_clean, I_filtered)

        # Step 4: Calculate residual RMS (using clean data)
        residual_rms = calculate_residual_rms(I_clean, I_filtered)

        # Step 5: Remove duplicate voltage values by averaging current at duplicate points
        V_unique, unique_indices = np.unique(V_clean, return_inverse=True)
        I_filtered_unique = np.array([I_filtered[unique_indices == i].mean()
                                      for i in range(len(V_unique))])

        # Step 6: Create smooth arrays for plotting (500 points)
        V_smooth = np.linspace(V_unique.min(), V_unique.max(), 500)

        # Interpolate filtered current onto smooth voltage grid using deduplicated data
        interp_func = interp1d(V_unique, I_filtered_unique, kind='linear',
                               bounds_error=False, fill_value='extrapolate')
        I_smooth = interp_func(V_smooth)

        # Step 7: Decompose filtered current into symmetric and asymmetric components
        I_filtered_sym, I_filtered_asym = decompose_symmetric_asymmetric(V_clean, I_filtered)
        I_smooth_sym, I_smooth_asym = decompose_symmetric_asymmetric(V_smooth, I_smooth)

        # Build and return filtered data
        return IVFilteredData(
            filename=extended_data.filename,
            filepath=extended_data.filepath,
            voltage=extended_data.voltage,
            current=extended_data.current,
            Hx=extended_data.Hx,
            Hy=extended_data.Hy,
            Hz=extended_data.Hz,
            H=extended_data.H,
            timestamp=extended_data.timestamp,

            outlier_mask=outlier_mask,
            n_outliers_removed=n_outliers,
            voltage_clean=V_clean,
            current_clean=I_clean,

            sigma=sigma,
            truncate=truncate,

            current_filtered=I_filtered,
            v_offset=v_offset,
            residual_rms=residual_rms,

            voltage_smooth=V_smooth,
            current_smooth=I_smooth,

            current_filtered_sym=I_filtered_sym,
            current_filtered_asym=I_filtered_asym,
            current_smooth_sym=I_smooth_sym,
            current_smooth_asym=I_smooth_asym,
        )

    except Exception as e:
        raise RuntimeError(f"Gaussian filtering failed: {str(e)}")


def save_filter_plots(filtered_data: IVFilteredData,
                     save_path: str,
                     filename: Optional[str] = None,
                     figsize: tuple = (12, 10),
                     dpi: int = 300) -> str:
    """
    Save plots of IV data with Gaussian filter and residuals.

    Parameters:
        filtered_data: IVFilteredData object with all filter results
        save_path: Directory path to save the plot
        filename: Custom filename (default: auto-generated from timestamp)
        figsize: Figure size (width, height) in inches
        dpi: Resolution in dots per inch

    Returns:
        Full path to saved plot
    """
    # Create save directory if it doesn't exist
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename if not provided
    if filename is None:
        filename = f"IV_gaussian_{filtered_data.timestamp}_H{filtered_data.H:.4f}T.png"

    full_path = save_dir / filename

    # Create plot with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))

    # Top plot: Raw vs Filtered IV curve with outliers highlighted
    # Plot clean data points
    clean_mask = ~filtered_data.outlier_mask
    ax1.plot(filtered_data.voltage[clean_mask], filtered_data.current[clean_mask] * 1e6, 'o',
             markersize=4, label='Clean Data', alpha=0.5, color='blue')

    # Highlight outliers if any
    if filtered_data.n_outliers_removed > 0:
        ax1.plot(filtered_data.voltage[filtered_data.outlier_mask],
                filtered_data.current[filtered_data.outlier_mask] * 1e6, 'x',
                markersize=8, label=f'Outliers ({filtered_data.n_outliers_removed})',
                color='red', markeredgewidth=2)

    # Plot filtered curve
    ax1.plot(filtered_data.voltage_smooth, filtered_data.current_smooth * 1e6, '-',
             linewidth=2.5, label=f'Gaussian Filter (σ={filtered_data.sigma})', color='darkgreen')

    # Mark voltage offset if valid
    if not np.isnan(filtered_data.v_offset):
        ax1.axvline(filtered_data.v_offset, color='orange', linestyle='--',
                    linewidth=2, label=f'V_offset = {filtered_data.v_offset:.4f} V')

    ax1.set_xlabel('Voltage (V)', fontsize=12)
    ax1.set_ylabel('Current (µA)', fontsize=12)
    ax1.set_title(
        f'Gaussian Filtered IV Curve (H = {filtered_data.H:.4f} T)\n'
        f'σ = {filtered_data.sigma:.2f}, V_offset = {filtered_data.v_offset:.4f} V\n'
        f'Outliers removed: {filtered_data.n_outliers_removed}/{len(filtered_data.voltage)}',
        fontsize=12,
    )
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)

    # Middle plot: Residuals (Clean - Filtered)
    residuals = filtered_data.current_clean - filtered_data.current_filtered
    ax2.plot(filtered_data.voltage_clean, residuals * 1e6, 'o-',
             markersize=4, linewidth=1, color='purple', alpha=0.6)
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.8)

    ax2.set_xlabel('Voltage (V)', fontsize=12)
    ax2.set_ylabel('Residual (µA)', fontsize=12)
    ax2.set_title(
        f'Residuals (Clean Data - Filtered)\n'
        f'RMS = {filtered_data.residual_rms * 1e6:.3f} µA',
        fontsize=12,
    )
    ax2.grid(True, alpha=0.3)

    # Bottom plot: Log scale view to see exponential behavior
    I_abs_clean = np.abs(filtered_data.current_clean)
    I_abs_smooth = np.abs(filtered_data.current_smooth)

    # Replace zeros with minimum positive value
    min_positive = np.min(I_abs_clean[I_abs_clean > 0]) if np.any(I_abs_clean > 0) else 1e-12
    I_abs_clean[I_abs_clean == 0] = min_positive
    I_abs_smooth[I_abs_smooth == 0] = min_positive

    ax3.semilogy(filtered_data.voltage_clean, I_abs_clean * 1e6, 'o',
                markersize=4, label='Clean Data', alpha=0.5, color='blue')
    ax3.semilogy(filtered_data.voltage_smooth, I_abs_smooth * 1e6, '-',
                linewidth=2.5, label='Filtered', color='darkgreen')

    ax3.set_xlabel('Voltage (V)', fontsize=12)
    ax3.set_ylabel('|Current| (µA)', fontsize=12)
    ax3.set_title('IV Curve (Log Scale) - Exponential Behavior', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(full_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    return str(full_path)


# Example usage
if __name__ == "__main__":

    # Example file path
    example_file = r"data/device 2/20251104191228TMR_CBS_10K_w_IV_82steps_H0.00T_phi_84.0to84.0_theta_0.0to0.0°_v0.datfolder/20251104193843_Hx_41.820000E-3_T_Hy_-150.000000E-6_T_Hz_397.900000E-3_T_decreasing.txt.dat"

    # Load and parse data
    print("Loading data...")
    iv_data = load_iv_data(example_file)
    extended_data = data_parser_IV(iv_data)

    # Apply Gaussian filter
    print("Applying Gaussian filter...")
    filtered_data = filter_iv_data(extended_data, sigma=1.5)

    # Display results
    print("\n=== Gaussian Filter Results ===")
    print(f"H = {filtered_data.H:.6e} T")
    print(f"Sigma = {filtered_data.sigma:.2f}")
    print(f"V_offset = {filtered_data.v_offset:.6f} V")
    print(f"Residual RMS = {filtered_data.residual_rms:.6e} A")

    # Save plots
    print("\nSaving plots...")
    save_path = "output/gaussian_filters"
    saved_file = save_filter_plots(filtered_data, save_path)
    print(f"Plots saved to: {saved_file}")
