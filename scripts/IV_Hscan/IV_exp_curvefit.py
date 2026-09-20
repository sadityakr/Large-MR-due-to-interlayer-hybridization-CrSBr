"""
IV Exponential Curve Fitting
Created: 05/11/2025

Fits IV data with two-level exponential models and provides error estimates.
Extends IVDataExtended with fit parameters and results.
"""
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
from scipy.optimize import curve_fit
from typing import Optional, Tuple
from .IV_H_scan_data_loader import IVDataExtended, load_iv_data, data_parser_IV


@dataclass
class IVAnalyzedData:
    """
    Extended data container with two-level exponential fit results.

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
        
        # Fit parameters: I = a*V*exp(b*|V|)
        a_fit: Fit parameter a (A/V)
        b_fit: Fit parameter b (V^-1)
        a_error: Standard error in a
        b_error: Standard error in b        
        # Second level fit parameters: I = a*(V-c)*exp(b*|V-c|)
        c_fit: Voltage offset parameter (V)
        c_error: Standard error in c
        r_squared_2: Goodness of fit for second level (R²)
        
        # Fit arrays
        voltage_fit: Voltage array used for fitting
        current_fit_1: First level fitted current values
        current_fit_2: Second level fitted current values
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
    # First level fit results (level 1)
    a_fit_1: float
    b_fit_1: float
    a_error_1: float
    b_error_1: float
    current_fit_1: np.ndarray

    # Second (final) level fit results
    a_fit: float
    b_fit: float
    c_fit: float
    a_error: float
    b_error: float
    c_error: float
    current_fit: np.ndarray

    # Common fit arrays and goodness-of-fit
    voltage_fit: np.ndarray
    r_squared_1: float
    r_squared_2: float


def iv_exponential(V: np.ndarray, a: float, b: float) -> np.ndarray:
    """First level exponential IV model: I = a*V*exp(b*|V|)"""
    return a * V * np.exp(b * np.abs(V))


def iv_exponential_w_voffset(V: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Second level exponential IV model: I = a*(V-c)*exp(b*|V-c|)"""
    return a * (V - c) * np.exp(b * np.abs(V - c))


def calculate_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R-squared value for model fit"""
    residuals = y_true - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)


def fit_iv_data(extended_data: IVDataExtended,
                p0_level1: Optional[list] = None,
                voltage_range: Optional[Tuple[float, float]] = None) -> IVAnalyzedData:
    """
    Perform two-level exponential fitting of IV data with error estimation.

    Parameters:
        extended_data: IVDataExtended object with IV data
        p0_level1: Initial guess for first level fit [a, b]. Default: [1e-7, 1.0]
        voltage_range: Optional (min_voltage, max_voltage) to restrict fitting range

    Returns:
        IVAnalyzedData object with all fit results and error estimates

    Raises:
        RuntimeError: If fitting fails
    
    """
    # Reasonable default for a: avoid division by zero
    V = extended_data.voltage
    I = extended_data.current

    if p0_level1 is None:
        # fallback guess - small conductance scale
        a_guess = 1e-6 if np.all(I == 0) else (np.max(np.abs(I)) / (np.max(np.abs(V)) + 1e-12))
        p0_level1 = [a_guess, 1.0]

    try:
        # First level fit: I = a*V*exp(b*|V|)
        popt1, pcov1 = curve_fit(iv_exponential, V, I, p0=p0_level1)
        a_fit_1, b_fit_1 = popt1

        # Parameter errors (if covariance returned)
        if pcov1 is not None:
            a_error_1, b_error_1 = np.sqrt(np.abs(np.diag(pcov1)))
        else:
            a_error_1 = b_error_1 = np.nan

        # First level R-squared
        I_pred_1 = iv_exponential(V, a_fit_1, b_fit_1)
        r_squared_1 = calculate_r_squared(I, I_pred_1)

        # Second level fit: I = a*(V-c)*exp(b*|V-c|)
        p0_level2 = [a_fit_1, b_fit_1, 0.0]
        popt2, pcov2 = curve_fit(iv_exponential_w_voffset, V, I, p0=p0_level2)
        a_fit, b_fit, c_fit = popt2

        if pcov2 is not None:
            a_error, b_error, c_error = np.sqrt(np.abs(np.diag(pcov2)))
        else:
            a_error = b_error = c_error = np.nan

        # Second level R-squared
        I_pred_2 = iv_exponential_w_voffset(V, a_fit, b_fit, c_fit)
        r_squared_2 = calculate_r_squared(I, I_pred_2)

        # Smooth voltage for plotting
        voltage_fit = np.linspace(V.min(), V.max(), 500)
        current_fit_1 = iv_exponential(voltage_fit, a_fit_1, b_fit_1)
        current_fit = iv_exponential_w_voffset(voltage_fit, a_fit, b_fit, c_fit)

        # Build and return analyzed data
        return IVAnalyzedData(
            filename=extended_data.filename,
            filepath=extended_data.filepath,
            voltage=extended_data.voltage,
            current=extended_data.current,
            Hx=extended_data.Hx,
            Hy=extended_data.Hy,
            Hz=extended_data.Hz,
            H=extended_data.H,
            timestamp=extended_data.timestamp,

            a_fit_1=a_fit_1,
            b_fit_1=b_fit_1,
            a_error_1=a_error_1,
            b_error_1=b_error_1,
            current_fit_1=current_fit_1,

            a_fit=a_fit,
            b_fit=b_fit,
            c_fit=c_fit,
            a_error=a_error,
            b_error=b_error,
            c_error=c_error,
            current_fit=current_fit,

            voltage_fit=voltage_fit,
            r_squared_1=r_squared_1,
            r_squared_2=r_squared_2,
        )

    except Exception as e:
        raise RuntimeError(f"Curve fitting failed: {str(e)}")


def save_fit_plots(analyzed_data: IVAnalyzedData,
                save_path: str,
                   filename: Optional[str] = None,
                   figsize: tuple = (12, 10),
                   dpi: int = 300) -> str:
    """
    Save plots of IV data with both levels of exponential fits.

    Parameters:
        analyzed_data: IVAnalyzedData object with all fit results
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
        filename = f"IV_fits_{analyzed_data.timestamp}_H{analyzed_data.H:.4f}T.png"

    full_path = save_dir / filename

    # Create plot with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    # First level fit plot
    ax1.plot(analyzed_data.voltage, analyzed_data.current * 1e6, 'o',
         markersize=6, label='Data', alpha=0.6)
    ax1.plot(analyzed_data.voltage_fit, analyzed_data.current_fit_1 * 1e6, 'r-',
         linewidth=2.5, label='Fit: I = a·V·exp(b·|V|)')

    ax1.set_xlabel('Voltage (V)', fontsize=12)
    ax1.set_ylabel('Current (µA)', fontsize=12)
    ax1.set_title(
    f'First Level Fit (H = {analyzed_data.H:.4f} T)\n'
    f'a = ({analyzed_data.a_fit_1:.2e} ± {analyzed_data.a_error_1:.2e}) A/V\n'
    f'b = ({analyzed_data.b_fit_1:.4f} ± {analyzed_data.b_error_1:.4f}) V⁻¹\n'
    f'R² = {analyzed_data.r_squared_1:.4f}',
    fontsize=12,
    )
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Second level (final) fit plot
    ax2.plot(analyzed_data.voltage, analyzed_data.current * 1e6, 'o',
         markersize=6, label='Data', alpha=0.6)
    ax2.plot(analyzed_data.voltage_fit, analyzed_data.current_fit * 1e6, 'g-',
         linewidth=2.5, label='Fit: I = a·(V-c)·exp(b·|V-c|)')

    ax2.set_xlabel('Voltage (V)', fontsize=12)
    ax2.set_ylabel('Current (µA)', fontsize=12)
    ax2.set_title(
    f'Second Level Fit\n'
    f'a = ({analyzed_data.a_fit:.2e} ± {analyzed_data.a_error:.2e}) A/V\n'
    f'b = ({analyzed_data.b_fit:.4f} ± {analyzed_data.b_error:.4f}) V⁻¹\n'
    f'c = ({analyzed_data.c_fit:.4f} ± {analyzed_data.c_error:.4f}) V\n'
    f'R² = {analyzed_data.r_squared_2:.4f}',
    fontsize=12,
    )
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

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

    # Perform fits
    print("Performing exponential fits...")
    analyzed_data = fit_iv_data(extended_data)

    # Display results for first-level fit
    print("\n=== First Level Fit Results ===")
    print(f"H = {analyzed_data.H:.6e} T")
    print(f"a = ({analyzed_data.a_fit_1:.6e} ± {analyzed_data.a_error_1:.6e}) A/V")
    print(f"b = ({analyzed_data.b_fit_1:.6f} ± {analyzed_data.b_error_1:.6f}) V⁻¹")
    print(f"R² = {analyzed_data.r_squared_1:.6f}")

    # Display results for second (final) level fit
    print("\n=== Second Level (Final) Fit Results ===")
    print(f"a = ({analyzed_data.a_fit:.6e} ± {analyzed_data.a_error:.6e}) A/V")
    print(f"b = ({analyzed_data.b_fit:.6f} ± {analyzed_data.b_error:.6f}) V⁻¹")
    print(f"c = ({analyzed_data.c_fit:.6f} ± {analyzed_data.c_error:.6f}) V")
    print(f"R² = {analyzed_data.r_squared_2:.6f}")

    # Save plots
    print("\nSaving plots...")
    save_path = "output/fits"
    saved_file = save_fit_plots(analyzed_data, save_path)
    print(f"Plots saved to: {saved_file}")
