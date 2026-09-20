"""
Batch Processing Script for All IV_H_scan Data

This script processes all temperature folders in b_scans and c_scans,
applying Gaussian filter analysis and creating dataframes.

Usage:
    python scripts/batch_process_all_scans.py --scan-type b_scans
    python scripts/batch_process_all_scans.py --scan-type c_scans
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.IV_Hscan_gaussian import (
    process_folder,
    build_dataframe,
    save_dataframe
)


def parse_folder_metadata(folder_name):
    """Extract temperature and timestamp from folder name."""
    parts = folder_name.split('_')

    # Extract temperature (look for pattern like "10K", "20K", etc.)
    temperature = None
    timestamp = None

    for i, part in enumerate(parts):
        if part.endswith('K'):
            try:
                temperature = int(part[:-1])  # Remove 'K' and convert to int
            except ValueError:
                pass

    # Extract timestamp (first part before first underscore if it's digits)
    if parts[0].isdigit():
        timestamp = parts[0]

    return temperature, timestamp


def process_scan_type(scan_type, sigma=1.5, verbose=True):
    """
    Process all temperature folders for a given scan type.

    Parameters:
    -----------
    scan_type : str
        Either 'b_scans' or 'c_scans'
    sigma : float
        Gaussian filter sigma parameter
    verbose : bool
        Print progress messages
    """
    # Define paths
    data_dir = PROJECT_ROOT / "data" / "device 2" / scan_type / "IV_H_scans"
    output_dir = PROJECT_ROOT / "output" / "IV_H_scans" / "dataframes" / scan_type

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        return

    # Get all folders in the directory
    folders = [f for f in data_dir.iterdir() if f.is_dir()]

    if verbose:
        print("="*70)
        print(f"BATCH PROCESSING: {scan_type}")
        print("="*70)
        print(f"Data directory: {data_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Found {len(folders)} folders to process")
        print()

    # Process each folder
    results = []
    for i, folder_path in enumerate(folders, 1):
        folder_name = folder_path.name
        temperature, timestamp = parse_folder_metadata(folder_name)

        if verbose:
            print(f"\n[{i}/{len(folders)}] Processing: {folder_name}")
            print(f"  Temperature: {temperature} K")
            print(f"  Timestamp: {timestamp}")

        try:
            # Process folder with Gaussian filter
            filtered_dict, metadata = process_folder(
                folder_path=str(folder_path),
                sigma=sigma,
                verbose=False  # Don't print details for each file
            )

            # Override temperature from metadata if available
            if temperature is None and 'temperature' in metadata:
                temperature = metadata['temperature']

            # Build DataFrame
            df = build_dataframe(filtered_dict, temperature=temperature)

            # Create output filename
            if timestamp and temperature:
                output_filename = f"IV_gaussian_{temperature}K_{timestamp}"
            elif temperature:
                output_filename = f"IV_gaussian_{temperature}K"
            else:
                output_filename = f"IV_gaussian_{folder_name}"

            output_path = output_dir / output_filename

            # Save DataFrame
            save_dataframe(df, output_path, format='pickle', save_csv_scalars=True)

            if verbose:
                print(f"  [OK] Processed {len(df)} IV scans")
                print(f"  [OK] Saved to: {output_filename}.pkl")

            results.append({
                'folder': folder_name,
                'temperature': temperature,
                'num_scans': len(df),
                'output_file': f"{output_filename}.pkl",
                'success': True
            })

        except Exception as e:
            print(f"  [ERROR] processing {folder_name}: {str(e)}")
            results.append({
                'folder': folder_name,
                'temperature': temperature,
                'success': False,
                'error': str(e)
            })

    # Print summary
    if verbose:
        print("\n" + "="*70)
        print("BATCH PROCESSING SUMMARY")
        print("="*70)
        successful = sum(1 for r in results if r['success'])
        print(f"Total folders: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(results) - successful}")
        print()

        print("Processed temperatures:")
        for r in sorted(results, key=lambda x: x.get('temperature', 999) if x.get('temperature') else 999):
            if r['success']:
                print(f"  {r['temperature']}K: {r['num_scans']} scans → {r['output_file']}")

        print("="*70)

    return results


def main():
    parser = argparse.ArgumentParser(description='Batch process IV_H_scan data')
    parser.add_argument('--scan-type', type=str, required=True,
                        choices=['b_scans', 'c_scans'],
                        help='Type of scan to process (b_scans or c_scans)')
    parser.add_argument('--sigma', type=float, default=1.5,
                        help='Gaussian filter sigma parameter (default: 1.5)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')

    args = parser.parse_args()

    # Run batch processing
    results = process_scan_type(
        scan_type=args.scan_type,
        sigma=args.sigma,
        verbose=not args.quiet
    )

    # Exit with error code if any processing failed
    if any(not r['success'] for r in results):
        sys.exit(1)


if __name__ == '__main__':
    main()
