"""
Data loader for Vector Scan measurements
Reads all columns from vector scan .dat files with metadata extraction
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
import re
from pathlib import Path


@dataclass
class VectorScan_data:
    """Data class to store vector scan measurement results"""
    file_path: str
    # Time data
    date: np.ndarray              # Date strings
    time: np.ndarray              # Time strings
    # Temperature data
    T_cryo: np.ndarray           # Cryostat temperature in K
    T_sample: np.ndarray         # Sample temperature in K
    # Magnetic field components
    Hx: np.ndarray               # X-component in Tesla
    Hy: np.ndarray               # Y-component in Tesla
    Hz: np.ndarray               # Z-component in Tesla
    H_magnitude: np.ndarray      # Field magnitude in Tesla
    phi: np.ndarray              # Phi angle in degrees
    theta: np.ndarray            # Theta angle in degrees
    # Lock-in measurements (5 channels, 2 readings each)
    L1_Ch1: np.ndarray
    L1_Ch2: np.ndarray
    L2_Ch1: np.ndarray
    L2_Ch2: np.ndarray
    L3_Ch1: np.ndarray
    L3_Ch2: np.ndarray
    L4_Ch1: np.ndarray
    L4_Ch2: np.ndarray
    L5_Ch1: np.ndarray
    L5_Ch2: np.ndarray
    # Metadata (extracted from filename or file)
    temperature: float = None     # type: ignore # Target temperature in K
    num_steps: int = None         # pyright: ignore[reportAssignmentType] # Number of scan steps
    scan_type: str = None         # type: ignore # "vector-scan"
    sample_name: str = None       # type: ignore # From file header
    user_name: str = None         # type: ignore # From file header
    # Calculated fields
    H: np.ndarray = None          # type: ignore # Signed field (magnitude with sign of dominant component)
    # TMR calculations
    TMR_pos: float = None         # type: ignore # TMR for L2 (positive)
    TMR_neg: float = None         # type: ignore # TMR for L4 (negative)
    TMR_average: float = None     # type: ignore # TMR for L5 (average)
    # R_norm calculations
    R_norm: np.ndarray = None     # type: ignore # Normalized resistance: (L5_Ch1 - min)/min
    R_norm_error: np.ndarray = None  # type: ignore # Error in R_norm


class VectorScanReader:
    """Class for reading vector scan data from .dat files"""
    
    def __init__(self, file_path: str):
        """
        Initialize with .dat file path and process the data
        
        Args:
            file_path: Path to the vector scan .dat file
        """
        self.file_path = file_path
        self.df = self._load_raw_data()
        self.temperature = self._read_temperature()
        self.scan_metadata = self._read_scan_metadata()
        self.header_metadata = self._read_header_metadata()
        self.vectorscan_data = self._process_data()
    
    def _load_raw_data(self) -> pd.DataFrame:
        """
        Load raw data from file, skipping header lines
        
        Returns:
            DataFrame with all 20 columns
        """
        # Define column names explicitly to avoid issues with trailing tabs
        column_names = [
            'date', 'time', 'Tcryo (K)', 'Tsample (K)', 
            'Hx (T)', 'Hy (T)', 'Hz (T)', '|H|', 'phi', 'theta',
            'L1_Ch1', 'L1_Ch2', 'L2_Ch1', 'L2_Ch2', 
            'L3_Ch1', 'L3_Ch2', 'L4_Ch1', 'L4_Ch2', 
            'L5_ Ch1', 'L5_Ch2'
        ]
        
        # Read the file to find where data starts (after header lines with !)
        with open(self.file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Find the column header line (first non-comment line with actual tab-separated content)
        header_line = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Look for line that doesn't start with !, has content, and contains tabs
            if not stripped.startswith('!') and stripped and '\t' in stripped:
                # This is the column header line
                header_line = i
                break
        
        # Read data using pandas, skipping header and using explicit column names
        df = pd.read_csv(
            self.file_path,
            sep='\t',
            skiprows=header_line + 1,  # Skip to first data row (after column headers)
            names=column_names,  # Use explicit column names
            na_values=['NA', 'N/A', 'nan', 'NaN', ''],
            skipinitialspace=True,
            encoding='utf-8',
            usecols=range(20)  # Only read first 20 columns (ignore trailing empty columns)
        )
        
        # Remove any completely empty rows
        df = df.dropna(how='all', axis=0)
        
        return df
    
    def _read_temperature(self) -> float:
        """
        Extract temperature from filename
        
        Filename format: 010K_20251006233100_vector-scan_...
        
        Returns:
            Temperature in Kelvin
        """
        filename = Path(self.file_path).name
        
        # Match pattern like "010K_" at the start
        t_match = re.search(r'^(\d+)K_', filename)
        if t_match:
            return float(t_match.group(1))
        else:
            raise ValueError(f"Could not parse temperature from filename: {filename}")
    
    def _read_scan_metadata(self) -> dict:
        """
        Extract scan metadata from filename
        
        Returns:
            Dictionary with scan parameters (num_steps, field_value, phi_range, theta_range)
        """
        filename = Path(self.file_path).name
        metadata = {}
        
        # Extract number of steps: "82steps"
        steps_match = re.search(r'(\d+)steps', filename)
        if steps_match:
            metadata['num_steps'] = int(steps_match.group(1))
        
        # Extract field value: "H-0.80T" or "H0.00T"
        field_match = re.search(r'H(-?\d+\.?\d*)T', filename)
        if field_match:
            metadata['field_value'] = float(field_match.group(1))
        
        # Extract phi range: "phi_92.0to92.0°"
        phi_match = re.search(r'phi_(-?\d+\.?\d*)to(-?\d+\.?\d*)°', filename)
        if phi_match:
            metadata['phi_start'] = float(phi_match.group(1))
            metadata['phi_end'] = float(phi_match.group(2))
        
        # Extract theta range: "theta_0.0to0.0°"
        theta_match = re.search(r'theta_(-?\d+\.?\d*)to(-?\d+\.?\d*)°', filename)
        if theta_match:
            metadata['theta_start'] = float(theta_match.group(1))
            metadata['theta_end'] = float(theta_match.group(2))
        
        # Extract scan type
        if 'vector-scan' in filename:
            metadata['scan_type'] = 'vector-scan'
        
        return metadata
    
    def _read_header_metadata(self) -> dict:
        """
        Parse header lines from file to extract metadata
        
        Returns:
            Dictionary with metadata (sample_name, user_name, date, time)
        """
        metadata = {}
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Parse header lines (lines starting with !)
            for line in lines:
                line = line.strip()
                if not line.startswith('!'):
                    break
                
                # Extract sample name (line with "Sample Name:")
                if 'Sample Name:' in line:
                    parts = line.split('Sample Name:')
                    if len(parts) > 1:
                        metadata['sample_name'] = parts[1].strip()
                
                # Extract user name (line with "User Name:")
                if 'User Name:' in line:
                    parts = line.split('User Name:')
                    if len(parts) > 1:
                        metadata['user_name'] = parts[1].strip()
                
                # Extract date (line with "Date:")
                if line.startswith('! Date:'):
                    parts = line.split('Date:')
                    if len(parts) > 1:
                        metadata['measurement_date'] = parts[1].strip()
                
                # Extract time (line with "Time:")
                if line.startswith('!Time:'):
                    parts = line.split('Time:')
                    if len(parts) > 1:
                        metadata['measurement_time'] = parts[1].strip()
        
        except Exception as e:
            print(f"Warning: Could not fully parse header metadata: {e}")
        
        return metadata
    
    def _calculate_signed_H(self, Hx: np.ndarray, Hy: np.ndarray, Hz: np.ndarray, 
                           H_magnitude: np.ndarray) -> np.ndarray:
        """
        Calculate signed magnetic field H
        Sign is determined by the component (Hx, Hy, or Hz) with largest magnitude
        
        Args:
            Hx: X-component of field
            Hy: Y-component of field
            Hz: Z-component of field
            H_magnitude: Magnitude of field
        
        Returns:
            Signed field H
        """
        H_signed = np.zeros_like(H_magnitude)
        
        for i in range(len(H_magnitude)):
            # Find which component has the largest magnitude
            abs_components = [np.abs(Hx[i]), np.abs(Hy[i]), np.abs(Hz[i])]
            max_idx = np.argmax(abs_components)
            
            # Get the sign from the dominant component
            if max_idx == 0:  # Hx is dominant
                sign = np.sign(Hx[i])
            elif max_idx == 1:  # Hy is dominant
                sign = np.sign(Hy[i])
            else:  # Hz is dominant
                sign = np.sign(Hz[i])
            
            # Apply sign to magnitude
            H_magnitude[i] = np.sqrt(Hx[i]**2 + Hy[i]**2 + Hz[i]**2)
            H_signed[i] = sign * H_magnitude[i]
        
        return H_signed
    
    def _calculate_TMR(self, L_data: np.ndarray) -> float:
        """
        Calculate TMR (Tunnel Magnetoresistance) for a given lock-in channel
        
        Logic:
        1. Calculate absolute value of L: abs_L
        2. Take mean of 10 largest values: L_max
        3. Take mean of 10 minimum values: L_min
        4. TMR = (L_max - L_min) / L_min
        
        Args:
            L_data: Lock-in channel data (e.g., L2_Ch1 or L4_Ch1)
        
        Returns:
            TMR value as float
        """
        if len(L_data) == 0:
            return np.nan
        
        # Calculate absolute value
        abs_L = np.abs(L_data)
        
        # Remove any NaN values
        abs_L = abs_L[~np.isnan(abs_L)]
        
        if len(abs_L) < 20:  # Need at least 20 points for 10 max + 10 min
            return np.nan
        
        # Sort the data
        sorted_L = np.sort(abs_L)
        
        # Get 10 largest values (last 10 in sorted array)
        L_max = np.mean(sorted_L[-10:])
        
        # Get 10 minimum values (first 10 in sorted array)
        L_min = np.mean(sorted_L[:10])
        
        # Calculate TMR
        if L_min == 0:
            return np.nan  # Avoid division by zero
        
        TMR = (L_max - L_min) / L_min
        
        return TMR # type: ignore
    
    def _calculate_R_norm(self, L5_Ch1: np.ndarray, L5_Ch2: np.ndarray) -> tuple:
        """
        Calculate normalized resistance R_norm and its error
        
        R_norm = (L5_Ch1 - min(L5_Ch1)) / min(L5_Ch1)
        
        Error propagation:
        For f = (x - x_min) / x_min, where x_min is constant:
        df/dx = 1/x_min
        So: error_f = error_x / |x_min|
        
        Args:
            L5_Ch1: L5 channel 1 data (resistance values)
            L5_Ch2: L5 channel 2 data (error values)
        
        Returns:
            Tuple of (R_norm, R_norm_error) as numpy arrays
        """
        if len(L5_Ch1) == 0 or len(L5_Ch2) == 0:
            return np.array([]), np.array([])
        
        # Remove any NaN values
        valid_mask = ~(np.isnan(L5_Ch1) | np.isnan(L5_Ch2))
        L5_Ch1_clean = L5_Ch1[valid_mask]
        L5_Ch2_clean = L5_Ch2[valid_mask]
        
        if len(L5_Ch1_clean) == 0:
            return np.array([]), np.array([])
        
        # Find minimum value
        L5_min = np.min(L5_Ch1_clean)
        
        # Avoid division by zero
        if L5_min == 0:
            return np.full_like(L5_Ch1_clean, np.nan), np.full_like(L5_Ch2_clean, np.nan)
        
        # Calculate R_norm
        R_norm = (L5_Ch1_clean - L5_min) / L5_min
        
        # Calculate error propagation
        # For f = (x - x_min) / x_min, df/dx = 1/x_min
        R_norm_error = L5_Ch2_clean / np.abs(L5_min)
        
        return R_norm, R_norm_error
    
    def _process_data(self) -> VectorScan_data:
        """
        Process raw DataFrame into VectorScan_data object
        
        Returns:
            VectorScan_data object with all columns and metadata
        """
        # Map DataFrame columns to expected names
        # Handle potential variations in column naming
        col_map = {
            'date': 'date',
            'time': 'time',
            'Tcryo (K)': 'T_cryo',
            'Tsample (K)': 'T_sample',
            'Hx (T)': 'Hx',
            'Hy (T)': 'Hy',
            'Hz (T)': 'Hz',
            '|H|': 'H_magnitude',
            'phi': 'phi',
            'theta': 'theta',
            'L1_Ch1': 'L1_Ch1',
            'L1_Ch2': 'L1_Ch2',
            'L2_Ch1': 'L2_Ch1',
            'L2_Ch2': 'L2_Ch2',
            'L3_Ch1': 'L3_Ch1',
            'L3_Ch2': 'L3_Ch2',
            'L4_Ch1': 'L4_Ch1',
            'L4_Ch2': 'L4_Ch2',
            'L5_ Ch1': 'L5_Ch1',  # Note the space in original
            'L5_Ch2': 'L5_Ch2'
        }
        
        # Extract data with proper type conversion
        data_dict = {}
        
        for df_col, data_col in col_map.items():
            if df_col in self.df.columns:
                if data_col in ['date', 'time']:
                    # Keep as strings
                    data_dict[data_col] = self.df[df_col].values.astype(str)
                else:
                    # Convert to float
                    data_dict[data_col] = self.df[df_col].values.astype(np.float64)
            else:
                # If column not found, create empty array
                print(f"Warning: Column '{df_col}' not found in data file")
                data_dict[data_col] = np.array([])
        
        # Calculate signed magnetic field H
        H_signed = self._calculate_signed_H(
            data_dict.get('Hx', np.array([])),
            data_dict.get('Hy', np.array([])),
            data_dict.get('Hz', np.array([])),
            data_dict.get('H_magnitude', np.array([]))
        )
        
        # Calculate TMR values
        TMR_pos = self._calculate_TMR(data_dict.get('L2_Ch1', np.array([])))
        TMR_neg = self._calculate_TMR(data_dict.get('L4_Ch1', np.array([])))
        TMR_average = self._calculate_TMR(data_dict.get('L5_Ch1', np.array([])))
        
        # Calculate R_norm values
        R_norm, R_norm_error = self._calculate_R_norm(
            data_dict.get('L5_Ch1', np.array([])),
            data_dict.get('L5_Ch2', np.array([]))
        )
        
        return VectorScan_data(
            file_path=self.file_path,
            date=data_dict.get('date', np.array([])),
            time=data_dict.get('time', np.array([])),
            T_cryo=data_dict.get('T_cryo', np.array([])),
            T_sample=data_dict.get('T_sample', np.array([])),
            Hx=data_dict.get('Hx', np.array([])),
            Hy=data_dict.get('Hy', np.array([])),
            Hz=data_dict.get('Hz', np.array([])),
            H_magnitude=data_dict.get('H_magnitude', np.array([])),
            H=H_signed,
            phi=data_dict.get('phi', np.array([])),
            theta=data_dict.get('theta', np.array([])),
            L1_Ch1=data_dict.get('L1_Ch1', np.array([])),
            L1_Ch2=data_dict.get('L1_Ch2', np.array([])),
            L2_Ch1=data_dict.get('L2_Ch1', np.array([])),
            L2_Ch2=data_dict.get('L2_Ch2', np.array([])),
            L3_Ch1=data_dict.get('L3_Ch1', np.array([])),
            L3_Ch2=data_dict.get('L3_Ch2', np.array([])),
            L4_Ch1=data_dict.get('L4_Ch1', np.array([])),
            L4_Ch2=data_dict.get('L4_Ch2', np.array([])),
            L5_Ch1=data_dict.get('L5_Ch1', np.array([])),
            L5_Ch2=data_dict.get('L5_Ch2', np.array([])),
            R_norm = R_norm,
            R_norm_error = R_norm_error,
            temperature=self.temperature,
            num_steps=self.scan_metadata.get('num_steps'), # type: ignore
            scan_type=self.scan_metadata.get('scan_type'), # type: ignore
            sample_name=self.header_metadata.get('sample_name'), # type: ignore
            user_name=self.header_metadata.get('user_name'), # type: ignore
            TMR_pos=TMR_pos,
            TMR_neg=TMR_neg,
            TMR_average=TMR_average,
        )


# Example usage
if __name__ == "__main__":
    # Load single file
    file_path = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "device 2" / "b_scans" / "H_scans" / "20251106153846b_scan_30K_92steps_H0.90T_phi_84.0to84.0_theta_0.0to0.0°_v0.dat")
    
    print("Loading vector scan data...")
    reader = VectorScanReader(file_path)
    data = reader.vectorscan_data
    
    print(f"\n{'='*70}")
    print("Vector Scan Data Summary")
    print(f"{'='*70}")
    
    print(f"\nMetadata:")
    print(f"  File: {Path(file_path).name}")
    print(f"  Temperature: {data.temperature:.1f} K")
    print(f"  Sample Name: {data.sample_name}")
    print(f"  User Name: {data.user_name}")
    print(f"  Scan Type: {data.scan_type}")
    print(f"  Number of Steps: {data.num_steps}")
    
    print(f"\nData Points: {len(data.Hx)}")
    
    print(f"\nTemperature:")
    print(f"  Cryostat: {data.T_cryo.mean():.2f} ± {data.T_cryo.std():.2f} K")
    print(f"  Sample: {data.T_sample.mean():.2f} K")
    
    print(f"\nMagnetic Field:")
    print(f"  Hx range: {data.Hx.min():.4f} to {data.Hx.max():.4f} T")
    print(f"  Hy range: {data.Hy.min():.4f} to {data.Hy.max():.4f} T")
    print(f"  Hz range: {data.Hz.min():.4f} to {data.Hz.max():.4f} T")
    print(f"  |H| range: {data.H_magnitude.min():.4f} to {data.H_magnitude.max():.4f} T")
    print(f"  H (signed) range: {data.H.min():.4f} to {data.H.max():.4f} T")
    print(f"  Phi: {data.phi.mean():.1f}°")
    print(f"  Theta: {data.theta.mean():.1f}°")
    
    print(f"\nLock-in Channel 1:")
    print(f"  L1_Ch1 range: {data.L1_Ch1.min():.3e} to {data.L1_Ch1.max():.3e}")
    print(f"  L1_Ch2 range: {data.L1_Ch2.min():.3e} to {data.L1_Ch2.max():.3e}")
    
    print(f"\nTMR Calculations:")
    print(f"  TMR_pos (L2): {data.TMR_pos:.6f}")
    print(f"  TMR_neg (L4): {data.TMR_neg:.6f}")
    print(f"  TMR_average (L5): {data.TMR_average:.6f}")
    
    print(f"\nR_norm Calculations:")
    if len(data.R_norm) > 0:
        print(f"  R_norm range: {data.R_norm.min():.6f} to {data.R_norm.max():.6f}")
        print(f"  R_norm mean: {data.R_norm.mean():.6f}")
        print(f"  R_norm_error mean: {data.R_norm_error.mean():.6e}")
    else:
        print(f"  R_norm: No data available")
    
    print(f"\n{'='*70}")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame({
        'Date': data.date[:5],
        'Time': data.time[:5],
        'T_cryo (K)': data.T_cryo[:5],
        'Hx (T)': data.Hx[:5],
        'Hy (T)': data.Hy[:5],
        'Hz (T)': data.Hz[:5],
        '|H| (T)': data.H_magnitude[:5],
        'L1_Ch1': data.L1_Ch1[:5],
    })
    
    print("\nFirst 5 data points:")
    print(summary_df.to_string(index=False))

