"""
IV H-Scan Analysis Module

Data loading and curve fitting tools for IV measurements with magnetic field scans.
"""

# Data loading and parsing
from .IV_H_scan_data_loader import IVDataLoader, IVData, IVDataExtended, load_iv_data, data_parser_IV

# Curve fitting
from .IV_exp_curvefit import IVAnalyzedData, fit_iv_data, save_fit_plots

# Batch processing
from .batch_processor import process_folder, parse_folder_name

# DataFrame tools
from .dataframe_builder import (
    build_dataframe,
    calculate_current_with_error,
    save_dataframe,
    load_dataframe,
    get_iv_at_field
)

__all__ = [
    # Data loader
    "IVDataLoader",
    "IVData",
    "IVDataExtended",
    "load_iv_data",
    "data_parser_IV",
    # Curve fitting
    "IVAnalyzedData",
    "fit_iv_data",
    "save_fit_plots",
    # Batch processing
    "process_folder",
    "parse_folder_name",
    # DataFrame tools
    "build_dataframe",
    "calculate_current_with_error",
    "save_dataframe",
    "load_dataframe",
    "get_iv_at_field",
]
