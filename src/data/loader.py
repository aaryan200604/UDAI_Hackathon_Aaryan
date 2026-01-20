"""
Data Loader Module
Handles loading and merging of UIDAI Aadhaar enrolment CSV files.
"""

import pandas as pd
import os
from typing import List, Optional
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import DATA_FILES, ORIGINAL_DATA_DIR, RAW_DATA_DIR


def load_single_file(filepath: str) -> pd.DataFrame:
    """
    Load a single CSV file with proper data types.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with parsed data
    """
    df = pd.read_csv(
        filepath,
        parse_dates=['date'],
        dayfirst=True,
        dtype={
            'state': 'category',
            'district': 'category',
            'pincode': 'str',
            'age_0_5': 'int64',
            'age_5_17': 'int64',
            'age_18_greater': 'int64'
        }
    )
    return df


def load_all_data(data_dir: Optional[str] = None, use_original: bool = True) -> pd.DataFrame:
    """
    Load and merge all Aadhaar enrolment CSV files.
    
    Args:
        data_dir: Directory containing CSV files (default: original data location)
        use_original: If True, use original data location from config
        
    Returns:
        Merged DataFrame with all enrolment data
    """
    if use_original:
        data_dir = ORIGINAL_DATA_DIR
    elif data_dir is None:
        data_dir = RAW_DATA_DIR
    
    dfs = []
    total_records = 0
    
    for filename in DATA_FILES:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            print(f"Loading {filename}...")
            df = load_single_file(filepath)
            dfs.append(df)
            total_records += len(df)
            print(f"  Loaded {len(df):,} records")
        else:
            print(f"Warning: File not found - {filepath}")
    
    if not dfs:
        raise FileNotFoundError(f"No data files found in {data_dir}")
    
    # Merge all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Sort by date and reset index
    merged_df = merged_df.sort_values('date').reset_index(drop=True)
    
    print(f"\nTotal records loaded: {total_records:,}")
    print(f"Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")
    print(f"States: {merged_df['state'].nunique()}")
    print(f"Districts: {merged_df['district'].nunique()}")
    
    return merged_df


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Generate summary statistics for the loaded data.
    
    Args:
        df: DataFrame with enrolment data
        
    Returns:
        Dictionary with summary statistics
    """
    # Calculate total enrolments per row
    df['total_enrolments'] = df['age_0_5'] + df['age_5_17'] + df['age_18_greater']
    
    summary = {
        'total_records': len(df),
        'date_range': {
            'start': df['date'].min().strftime('%Y-%m-%d'),
            'end': df['date'].max().strftime('%Y-%m-%d')
        },
        'states': df['state'].nunique(),
        'districts': df['district'].nunique(),
        'pincodes': df['pincode'].nunique(),
        'total_enrolments': {
            'age_0_5': df['age_0_5'].sum(),
            'age_5_17': df['age_5_17'].sum(),
            'age_18_greater': df['age_18_greater'].sum(),
            'total': df['total_enrolments'].sum()
        },
        'daily_stats': {
            'mean': df['total_enrolments'].mean(),
            'std': df['total_enrolments'].std(),
            'min': df['total_enrolments'].min(),
            'max': df['total_enrolments'].max()
        }
    }
    
    return summary


if __name__ == "__main__":
    # Test data loading
    print("Testing data loader...")
    df = load_all_data()
    print("\nData Summary:")
    summary = get_data_summary(df)
    for key, value in summary.items():
        print(f"  {key}: {value}")
