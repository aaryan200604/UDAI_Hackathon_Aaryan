"""
Data Preprocessor Module
Handles data cleaning, validation, and aggregation for UIDAI Aadhaar data.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the enrolment data by handling missing values and invalid entries.
    
    Args:
        df: Raw DataFrame with enrolment data
        
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Remove rows with missing critical values
    critical_cols = ['date', 'state', 'district']
    df_clean = df_clean.dropna(subset=critical_cols)
    
    # Fill missing age values with 0 (no enrolments)
    age_cols = ['age_0_5', 'age_5_17', 'age_18_greater']
    df_clean[age_cols] = df_clean[age_cols].fillna(0)
    
    # Ensure non-negative values for age columns
    for col in age_cols:
        df_clean[col] = df_clean[col].clip(lower=0)
    
    # Calculate total enrolments
    df_clean['total_enrolments'] = (
        df_clean['age_0_5'] + 
        df_clean['age_5_17'] + 
        df_clean['age_18_greater']
    )
    
    # Clean pincode - ensure it's a string and handle invalid values
    df_clean['pincode'] = df_clean['pincode'].astype(str).str.strip()
    df_clean.loc[df_clean['pincode'].str.len() != 6, 'pincode'] = 'UNKNOWN'
    
    # Standardize state and district names
    df_clean['state'] = df_clean['state'].astype(str).str.strip().str.title()
    df_clean['district'] = df_clean['district'].astype(str).str.strip().str.title()
    
    print(f"Cleaned data: {len(df_clean):,} records (removed {len(df) - len(df_clean):,})")
    
    return df_clean


def validate_data(df: pd.DataFrame) -> Tuple[bool, dict]:
    """
    Validate the data for consistency and completeness.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Tuple of (is_valid, validation_report)
    """
    issues = []
    warnings = []
    
    # Check for required columns
    required_cols = ['date', 'state', 'district', 'pincode', 
                     'age_0_5', 'age_5_17', 'age_18_greater']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    
    # Check for null values
    null_counts = df[required_cols].isnull().sum()
    if null_counts.any():
        warnings.append(f"Null values found: {null_counts[null_counts > 0].to_dict()}")
    
    # Check for negative values in age columns
    age_cols = ['age_0_5', 'age_5_17', 'age_18_greater']
    for col in age_cols:
        if (df[col] < 0).any():
            issues.append(f"Negative values found in {col}")
    
    # Check date range
    if df['date'].min() > df['date'].max():
        issues.append("Invalid date range")
    
    # Check for duplicate entries
    duplicates = df.duplicated(subset=['date', 'state', 'district', 'pincode']).sum()
    if duplicates > 0:
        warnings.append(f"Found {duplicates} duplicate entries")
    
    is_valid = len(issues) == 0
    
    report = {
        'is_valid': is_valid,
        'issues': issues,
        'warnings': warnings,
        'record_count': len(df),
        'date_range': (df['date'].min(), df['date'].max()),
        'states': df['state'].nunique(),
        'districts': df['district'].nunique()
    }
    
    return is_valid, report


def aggregate_by_district(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate enrolment data by district and date.
    
    Args:
        df: DataFrame with enrolment data
        
    Returns:
        Aggregated DataFrame by district
    """
    agg_df = df.groupby(['date', 'state', 'district']).agg({
        'age_0_5': 'sum',
        'age_5_17': 'sum',
        'age_18_greater': 'sum',
        'pincode': 'nunique'
    }).reset_index()
    
    agg_df = agg_df.rename(columns={'pincode': 'pincode_count'})
    
    # Calculate total and proportions
    agg_df['total_enrolments'] = (
        agg_df['age_0_5'] + agg_df['age_5_17'] + agg_df['age_18_greater']
    )
    
    agg_df['child_ratio'] = (
        (agg_df['age_0_5'] + agg_df['age_5_17']) / 
        agg_df['total_enrolments'].replace(0, np.nan)
    )
    
    agg_df['adult_ratio'] = (
        agg_df['age_18_greater'] / 
        agg_df['total_enrolments'].replace(0, np.nan)
    )
    
    return agg_df.sort_values(['date', 'state', 'district']).reset_index(drop=True)


def aggregate_by_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate enrolment data by state and date.
    
    Args:
        df: DataFrame with enrolment data
        
    Returns:
        Aggregated DataFrame by state
    """
    agg_df = df.groupby(['date', 'state']).agg({
        'age_0_5': 'sum',
        'age_5_17': 'sum',
        'age_18_greater': 'sum',
        'district': 'nunique',
        'pincode': 'nunique'
    }).reset_index()
    
    agg_df = agg_df.rename(columns={
        'district': 'district_count',
        'pincode': 'pincode_count'
    })
    
    # Calculate total and proportions
    agg_df['total_enrolments'] = (
        agg_df['age_0_5'] + agg_df['age_5_17'] + agg_df['age_18_greater']
    )
    
    agg_df['child_ratio'] = (
        (agg_df['age_0_5'] + agg_df['age_5_17']) / 
        agg_df['total_enrolments'].replace(0, np.nan)
    )
    
    agg_df['adult_ratio'] = (
        agg_df['age_18_greater'] / 
        agg_df['total_enrolments'].replace(0, np.nan)
    )
    
    return agg_df.sort_values(['date', 'state']).reset_index(drop=True)


def get_daily_totals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get daily total enrolments across all regions.
    
    Args:
        df: DataFrame with enrolment data
        
    Returns:
        DataFrame with daily totals
    """
    daily_df = df.groupby('date').agg({
        'age_0_5': 'sum',
        'age_5_17': 'sum',
        'age_18_greater': 'sum'
    }).reset_index()
    
    daily_df['total_enrolments'] = (
        daily_df['age_0_5'] + daily_df['age_5_17'] + daily_df['age_18_greater']
    )
    
    return daily_df.sort_values('date').reset_index(drop=True)


if __name__ == "__main__":
    # Test preprocessor
    from loader import load_all_data
    
    print("Testing preprocessor...")
    df = load_all_data()
    
    print("\nCleaning data...")
    df_clean = clean_data(df)
    
    print("\nValidating data...")
    is_valid, report = validate_data(df_clean)
    print(f"Validation: {'PASSED' if is_valid else 'FAILED'}")
    print(f"Report: {report}")
    
    print("\nAggregating by state...")
    state_df = aggregate_by_state(df_clean)
    print(f"State aggregation: {len(state_df)} records")
    print(state_df.head())
