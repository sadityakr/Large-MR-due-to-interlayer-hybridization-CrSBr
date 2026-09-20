"""
Batch processing module for IV H-scan measurements with Gaussian filtering.

This module processes all IV scans in a folder, filters them, and saves results.
"""

import re
from pathlib import Path
from typing import Dict, Tuple, Optional
from tqdm import tqdm

from ..IV_Hscan.IV_H_scan_data_loader import load_iv_data, data_parser_IV
from .IV_gaussian_filter import filter_iv_data, save_filter_plots, IVFilteredData


def parse_folder_name(folder_name: str) -> Dict[str, any]:
    """
    Extract metadata from folder name.

    Expected format:
    YYYYMMDDHHMMSSTMR_CBS_10K_w_IV_82steps_H0.00T_phi_84.0to84.0_theta_0.0to0.0°_v0.datfolder

    Parameters:
        folder_name: Folder name (not full path)

    Returns:
        Dictionary with keys: timestamp, temperature, num_steps, field_range, phi_range, theta_range
    """
    metadata = {}

    # Extract timestamp (first 14 digits)
    timestamp_match = re.match(r'^(\d{14})', folder_name)
    if timestamp_match:
        metadata['timestamp'] = timestamp_match.group(1)

    # Extract temperature (e.g., "10K")
    temp_match = re.search(r'CBS_(\d+(?:\.\d+)?)K', folder_name)
    if temp_match:
        metadata['temperature'] = float(temp_match.group(1))

    # Extract number of steps (e.g., "82steps")
    steps_match = re.search(r'(\d+)steps', folder_name)
    if steps_match:
        metadata['num_steps'] = int(steps_match.group(1))

    # Extract field range (e.g., "H0.00T")
    field_match = re.search(r'H([-+]?\d+(?:\.\d+)?)T', folder_name)
    if field_match:
        metadata['field_range'] = float(field_match.group(1))

    # Extract phi range (e.g., "phi_84.0to84.0")
    phi_match = re.search(r'phi_([-+]?\d+(?:\.\d+)?)to([-+]?\d+(?:\.\d+)?)', folder_name)
    if phi_match:
        metadata['phi_range'] = (float(phi_match.group(1)), float(phi_match.group(2)))

    # Extract theta range (e.g., "theta_0.0to0.0")
    theta_match = re.search(r'theta_([-+]?\d+(?:\.\d+)?)to([-+]?\d+(?:\.\d+)?)', folder_name)
    if theta_match:
        metadata['theta_range'] = (float(theta_match.group(1)), float(theta_match.group(2)))

    return metadata


def process_folder(folder_path: str,
                  output_base_dir: str = "output/gaussian_filters",
                  sigma: float = 1.5,
                  skip_existing: bool = False,
                  verbose: bool = True) -> Tuple[Dict[str, IVFilteredData], Dict]:
    """
    Process all IV scans in a folder with Gaussian filtering.

    This function:
    1. Finds all .dat files in the folder
    2. Processes each file: load → parse → filter → save plots
    3. Collects results in a dictionary

    Parameters:
        folder_path: Path to folder containing IV scan files
        output_base_dir: Base directory for saving filter plots (default: "output/gaussian_filters")
        sigma: Gaussian filter sigma value (default: 1.5)
        skip_existing: Skip files if filter plot already exists (default: False)
        verbose: Print progress messages (default: True)

    Returns:
        Tuple of (filtered_dict, metadata):
            - filtered_dict: Dict mapping timestamp → IVFilteredData
            - metadata: Dict with folder metadata (temperature, num_steps, etc.)

    Raises:
        FileNotFoundError: If folder_path doesn't exist
    """
    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # Parse folder name for metadata
    folder_name = folder_path.name
    metadata = parse_folder_name(folder_name)

    if verbose:
        print(f"\n{'='*70}")
        print(f"Processing folder: {folder_name}")
        print(f"{'='*70}")
        print(f"Metadata extracted:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        print()

    # Find all .dat files
    dat_files = list(folder_path.glob("*.dat"))

    if len(dat_files) == 0:
        print(f"Warning: No .dat files found in {folder_path}")
        return {}, metadata

    if verbose:
        print(f"Found {len(dat_files)} .dat files")
        print(f"Gaussian filter sigma: {sigma}")
        print()

    # Create output directory for filter plots
    output_path = Path(output_base_dir) / folder_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Process each file
    filtered_dict = {}
    failed_files = []
    skipped_files = []

    # Use tqdm for progress bar
    iterator = tqdm(dat_files, desc="Processing IV scans") if verbose else dat_files

    for file_path in iterator:
        try:
            # Check if we should skip this file
            if skip_existing:
                # Check if filter plot already exists (approximate check)
                potential_output = output_path / f"*{file_path.stem}*.png"
                if list(output_path.glob(potential_output.name)):
                    skipped_files.append(file_path.name)
                    continue

            # Step 1: Load data
            iv_data = load_iv_data(str(file_path))

            # Step 2: Parse metadata from filename
            extended_data = data_parser_IV(iv_data)

            # Step 3: Apply Gaussian filter
            filtered_data = filter_iv_data(extended_data, sigma=sigma)

            # Step 4: Save filter plots
            save_filter_plots(filtered_data, str(output_path))

            # Step 5: Store in dictionary using timestamp as key
            filtered_dict[extended_data.timestamp] = filtered_data

        except Exception as e:
            failed_files.append((file_path.name, str(e)))
            if verbose:
                print(f"\nError processing {file_path.name}: {e}")

    # Print summary
    if verbose:
        print(f"\n{'='*70}")
        print(f"PROCESSING SUMMARY")
        print(f"{'='*70}")
        print(f"Total files: {len(dat_files)}")
        print(f"Successfully processed: {len(filtered_dict)}")
        print(f"Failed: {len(failed_files)}")
        print(f"Skipped: {len(skipped_files)}")

        if failed_files:
            print(f"\nFailed files:")
            for filename, error in failed_files[:10]:  # Show first 10
                print(f"  - {filename}: {error}")
            if len(failed_files) > 10:
                print(f"  ... and {len(failed_files) - 10} more")

        print(f"\nFilter plots saved to: {output_path}")
        print(f"{'='*70}\n")

    # Add processing stats to metadata
    metadata['total_files'] = len(dat_files)
    metadata['successful'] = len(filtered_dict)
    metadata['failed'] = len(failed_files)
    metadata['skipped'] = len(skipped_files)
    metadata['failed_files'] = failed_files
    metadata['sigma'] = sigma

    return filtered_dict, metadata
