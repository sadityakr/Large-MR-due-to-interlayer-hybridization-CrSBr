"""
IV H-scan Gaussian Filter Analysis Package

This package provides tools for analyzing IV characteristic data from TMR measurements
using Gaussian filtering instead of exponential curve fitting.

Main Components:
    - IV_gaussian_filter: Core filtering functions and data structures
    - dataframe_builder: DataFrame construction and data management
    - batch_processor: Batch processing for entire folders

Example Usage:
    # Single file analysis
    from scripts.IV_Hscan_gaussian import load_iv_data, data_parser_IV, filter_iv_data

    iv_data = load_iv_data("path/to/file.dat")
    extended_data = data_parser_IV(iv_data)
    filtered_data = filter_iv_data(extended_data, sigma=1.5)

    # Batch processing
    from scripts.IV_Hscan_gaussian import process_folder, build_dataframe

    filtered_dict, metadata = process_folder("path/to/folder")
    df = build_dataframe(filtered_dict, temperature=10.0)
"""

# Import data loading from IV_Hscan package (shared infrastructure)
from ..IV_Hscan.IV_H_scan_data_loader import (
    IVData,
    IVDataExtended,
    load_iv_data,
    data_parser_IV
)

# Import Gaussian filter components
from .IV_gaussian_filter import (
    IVFilteredData,
    filter_iv_data,
    find_voltage_offset,
    calculate_residual_rms,
    save_filter_plots,
    decompose_symmetric_asymmetric,
    detect_outliers,
    remove_outliers
)

# Import DataFrame utilities
from .dataframe_builder import (
    build_dataframe,
    save_dataframe,
    load_dataframe,
    get_iv_at_field,
    get_current_at_voltage,
    get_asymmetric_current_at_voltage
)

# Import batch processing
from .batch_processor import (
    process_folder,
    parse_folder_name
)

__all__ = [
    # Data loading (from IV_Hscan)
    'IVData',
    'IVDataExtended',
    'load_iv_data',
    'data_parser_IV',

    # Gaussian filtering
    'IVFilteredData',
    'filter_iv_data',
    'find_voltage_offset',
    'calculate_residual_rms',
    'save_filter_plots',
    'decompose_symmetric_asymmetric',
    'detect_outliers',
    'remove_outliers',

    # DataFrame utilities
    'build_dataframe',
    'save_dataframe',
    'load_dataframe',
    'get_iv_at_field',
    'get_current_at_voltage',
    'get_asymmetric_current_at_voltage',

    # Batch processing
    'process_folder',
    'parse_folder_name',
]
