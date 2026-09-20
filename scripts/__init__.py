"""
CrSBr Data Analysis Scripts
Modular data loaders and analysis tools
"""

# Data loaders

from .data_loader_vector_scan import VectorScanReader, VectorScan_data

# IV H-scan analysis (in IV_Hscan subfolder)
from .IV_Hscan import (
    IVDataLoader, IVData, IVDataExtended, load_iv_data, data_parser_IV,
    IVAnalyzedData, fit_iv_data, save_fit_plots
)

# Notebook utilities
from .utils import setup_notebook

__all__ = [
    # Data loaders
    "VectorScanReader",
    "VectorScan_data",
    "IVDataLoader",
    "IVData",
    "IVDataExtended",
    # Analysis tools
    "IVAnalyzedData",
    "load_iv_data",
    "data_parser_IV",
    "fit_iv_data",
    "save_fit_plots",
    # Notebook utilities
    "setup_notebook",
]

__version__ = "0.1.0"