# IV H-Scan Gaussian Filter Analysis

This package provides a complete pipeline for analyzing IV characteristic data from TMR measurements using Gaussian filtering instead of exponential curve fitting.

## Overview

The Gaussian filter analysis scheme mirrors the structure of the exponential curve fitting approach (`IV_Hscan`), but uses Gaussian smoothing (σ=1.5) to filter IV curves and estimates voltage offset from the zero-crossing of the filtered current.

## Package Structure

```
scripts/IV_Hscan_gaussian/
├── __init__.py                  # Package initialization and exports
├── IV_gaussian_filter.py        # Core filtering functions and IVFilteredData class
├── dataframe_builder.py         # DataFrame construction and utilities
├── batch_processor.py           # Batch processing for entire folders
└── README.md                    # This file
```

## Key Components

### 1. IV_gaussian_filter.py

**IVFilteredData Class:**
- Extends `IVDataExtended` with filtered results
- Contains: sigma, v_offset, residual_rms
- Stores both raw and filtered current arrays

**Key Functions:**
- `filter_iv_data(extended_data, sigma=1.5)` - Apply Gaussian filter
- `find_voltage_offset(voltage, current_filtered)` - Find zero-crossing
- `calculate_residual_rms(current_raw, current_filtered)` - Quality metric
- `save_filter_plots(filtered_data, save_path)` - Save visualization plots

### 2. dataframe_builder.py

**Key Functions:**
- `build_dataframe(filtered_dict, temperature)` - Convert results to DataFrame
- `save_dataframe(df, output_path)` - Save as pickle/CSV
- `load_dataframe(input_path)` - Load saved DataFrame
- `get_iv_at_field(df, H_value, tolerance)` - Retrieve data at specific field
- `get_current_at_voltage(df, V_value, use_filtered)` - Extract I(H) at fixed V

### 3. batch_processor.py

**Key Functions:**
- `process_folder(folder_path, sigma=1.5, ...)` - Process all .dat files
- `parse_folder_name(folder_name)` - Extract metadata from folder names

## Usage Examples

### Single File Analysis

```python
from scripts.IV_Hscan_gaussian import (
    load_iv_data,
    data_parser_IV,
    filter_iv_data,
    save_filter_plots
)

# Load data
iv_data = load_iv_data("path/to/file.dat")
extended_data = data_parser_IV(iv_data)

# Apply Gaussian filter
filtered_data = filter_iv_data(extended_data, sigma=1.5)

# Display results
print(f"V_offset: {filtered_data.v_offset:.6f} V")
print(f"Residual RMS: {filtered_data.residual_rms:.6e} A")

# Save plots
save_filter_plots(filtered_data, "output/gaussian_filters")
```

### Batch Processing

```python
from scripts.IV_Hscan_gaussian import (
    process_folder,
    build_dataframe,
    save_dataframe
)

# Process entire folder
filtered_dict, metadata = process_folder(
    folder_path="path/to/folder",
    sigma=1.5,
    verbose=True
)

# Build DataFrame
df = build_dataframe(filtered_dict, temperature=10.0)

# Save results
save_dataframe(df, "output/dataframes/IV_gaussian_results")
```

### Extract I(H) Curves

```python
from scripts.IV_Hscan_gaussian import get_current_at_voltage

# Get current vs H at V=0.3V
I_vs_H = get_current_at_voltage(df, V_value=0.3, use_filtered=True)

# Plot
import matplotlib.pyplot as plt
plt.plot(I_vs_H['H'], I_vs_H['I_at_V'] * 1e6, 'o-')
plt.xlabel('H (T)')
plt.ylabel('Current (µA)')
plt.show()
```

## Notebooks

Two Jupyter notebooks demonstrate the complete workflow:

1. **`IV_H_scan_gaussian_analysis.ipynb`** - Single file analysis
   - Load and visualize raw data
   - Apply Gaussian filter
   - Display voltage offset and residual RMS
   - Compare raw vs filtered curves

2. **`IV_H_scan_gaussian_batch_analysis.ipynb`** - Batch processing
   - Process entire folder of measurements
   - Build and save DataFrame
   - Quality assessment via residual RMS
   - Voltage offset vs magnetic field analysis
   - I(H) curves at fixed voltages
   - TMR calculation

## Output Files

### Filter Plots
- Location: `output/gaussian_filters/<folder_name>/`
- Format: PNG (300 dpi)
- Filename: `IV_gaussian_{timestamp}_H{H}T.png`
- Content: 2-panel plot (filtered curve + residuals)

### DataFrames
- Location: `output/dataframes/`
- Formats:
  - `.pkl` - Full DataFrame with arrays (for Python analysis)
  - `.csv` - Scalar columns only (for spreadsheet viewing)

## DataFrame Columns

**Metadata:**
- timestamp, filename, H, Hx, Hy, Hz, temperature

**Filter Parameters:**
- sigma (default: 1.5)
- v_offset (V) - voltage where I_filtered = 0
- residual_rms (A) - RMS difference between raw and filtered

**Data Arrays:**
- voltage, current - raw measurements
- current_filtered - filtered at original voltage points
- voltage_smooth, current_smooth - smooth arrays for plotting (500 pts)

## Key Features

✓ **Gaussian Filtering**: Smooth IV curves with configurable sigma (default 1.5)
✓ **Voltage Offset Detection**: Automatic zero-crossing detection with linear interpolation
✓ **Quality Metrics**: Residual RMS for assessing filtering quality
✓ **Batch Processing**: Process entire folders with progress tracking
✓ **DataFrame Integration**: Organize results for easy analysis
✓ **Modular Design**: Small focused scripts (<200 lines each)
✓ **Shared Infrastructure**: Reuses data loader from IV_Hscan package

## Comparison with Exponential Fitting

| Feature | Gaussian Filter | Exponential Fit |
|---------|----------------|-----------------|
| Method | scipy.ndimage.gaussian_filter1d | scipy.optimize.curve_fit |
| Parameters | sigma, truncate | a, b, c (fit params) |
| V_offset | Zero-crossing detection | c parameter |
| Quality | Residual RMS | R² (goodness of fit) |
| Speed | Fast (no optimization) | Slower (iterative fitting) |
| Assumptions | None | Exponential IV model |

## Dependencies

- numpy
- pandas
- matplotlib
- scipy (ndimage, interpolate)
- tqdm (progress bars)
- pathlib

## Testing

Run the test scripts to verify installation:

```bash
# Test single file processing
python test_gaussian_filter.py

# Test batch processing
python test_batch_gaussian.py
```

## Notes

- The Gaussian filter sigma is fixed at 1.5 (optimal for typical IV noise)
- V_offset is found by linear interpolation at zero-crossing
- If multiple zero-crossings exist, returns the one closest to V=0
- Residual RMS provides a measure of noise reduction
- All data (raw and filtered) is preserved for verification

## References

This package follows the modular design principles outlined in `CLAUDE.md`:
- Small, focused scripts (<200 lines)
- Lightweight data readers
- Notebooks as orchestrators
- Reusable components
