"""
Demographic Features Module
Calculates age-based indicators and demographic patterns.
"""

import pandas as pd
import numpy as np
from typing import Optional


def calculate_age_imbalance(df: pd.DataFrame,
                            expected_ratios: Optional[dict] = None) -> pd.DataFrame:
    """
    Calculate demographic imbalance score based on deviation from expected age distribution.
    
    Args:
        df: DataFrame with age group columns
        expected_ratios: Expected ratios for each age group (default based on India demographics)
        
    Returns:
        DataFrame with imbalance score added
    """
    if expected_ratios is None:
        # Approximate expected ratios based on India demographics
        expected_ratios = {
            'age_0_5': 0.10,    # ~10% children 0-5
            'age_5_17': 0.25,   # ~25% children 5-17
            'age_18_greater': 0.65  # ~65% adults
        }
    
    df = df.copy()
    
    # Calculate total if not present
    if 'total_enrolments' not in df.columns:
        df['total_enrolments'] = df['age_0_5'] + df['age_5_17'] + df['age_18_greater']
    
    # Calculate actual ratios
    df['ratio_0_5'] = df['age_0_5'] / df['total_enrolments'].replace(0, np.nan)
    df['ratio_5_17'] = df['age_5_17'] / df['total_enrolments'].replace(0, np.nan)
    df['ratio_18_greater'] = df['age_18_greater'] / df['total_enrolments'].replace(0, np.nan)
    
    # Calculate deviation from expected ratios
    df['dev_0_5'] = abs(df['ratio_0_5'] - expected_ratios['age_0_5'])
    df['dev_5_17'] = abs(df['ratio_5_17'] - expected_ratios['age_5_17'])
    df['dev_18_greater'] = abs(df['ratio_18_greater'] - expected_ratios['age_18_greater'])
    
    # Composite imbalance score (mean absolute deviation)
    df['demographic_imbalance'] = (df['dev_0_5'] + df['dev_5_17'] + df['dev_18_greater']) / 3
    
    return df


def calculate_adult_child_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate adult to child enrolment ratio.
    
    Args:
        df: DataFrame with age group columns
        
    Returns:
        DataFrame with adult:child ratio added
    """
    df = df.copy()
    
    # Total children (0-17)
    df['total_children'] = df['age_0_5'] + df['age_5_17']
    
    # Adult to child ratio
    df['adult_child_ratio'] = df['age_18_greater'] / df['total_children'].replace(0, np.nan)
    
    # Cap extreme values
    df['adult_child_ratio'] = df['adult_child_ratio'].clip(upper=100)
    
    # Also calculate child dominance indicator (high = more children than expected)
    df['child_dominance'] = np.where(df['adult_child_ratio'] < 0.5, 'high',
                                     np.where(df['adult_child_ratio'] > 2, 'low', 'normal'))
    
    return df


def calculate_age_concentration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate concentration of enrolments in specific age groups.
    
    Args:
        df: DataFrame with age group columns
        
    Returns:
        DataFrame with concentration metrics added
    """
    df = df.copy()
    
    if 'total_enrolments' not in df.columns:
        df['total_enrolments'] = df['age_0_5'] + df['age_5_17'] + df['age_18_greater']
    
    # Calculate ratios
    df['ratio_0_5'] = df['age_0_5'] / df['total_enrolments'].replace(0, np.nan)
    df['ratio_5_17'] = df['age_5_17'] / df['total_enrolments'].replace(0, np.nan)
    df['ratio_18_greater'] = df['age_18_greater'] / df['total_enrolments'].replace(0, np.nan)
    
    # Dominant age group
    df['dominant_age_group'] = np.where(
        (df['ratio_0_5'] > df['ratio_5_17']) & (df['ratio_0_5'] > df['ratio_18_greater']),
        '0-5',
        np.where(
            (df['ratio_5_17'] > df['ratio_0_5']) & (df['ratio_5_17'] > df['ratio_18_greater']),
            '5-17',
            '18+'
        )
    )
    
    # Concentration score (Herfindahl-like index - higher = more concentrated)
    df['age_concentration'] = (
        df['ratio_0_5']**2 + df['ratio_5_17']**2 + df['ratio_18_greater']**2
    )
    
    return df


def detect_demographic_anomalies(df: pd.DataFrame,
                                 imbalance_threshold: float = 0.15) -> pd.DataFrame:
    """
    Detect demographic anomalies based on unusual age distributions.
    
    Args:
        df: DataFrame with demographic features
        imbalance_threshold: Threshold for flagging imbalance
        
    Returns:
        DataFrame with anomaly flags added
    """
    df = df.copy()
    
    # Ensure we have the imbalance score
    if 'demographic_imbalance' not in df.columns:
        df = calculate_age_imbalance(df)
    
    if 'adult_child_ratio' not in df.columns:
        df = calculate_adult_child_ratio(df)
    
    # Flag high imbalance
    df['high_imbalance'] = df['demographic_imbalance'] > imbalance_threshold
    
    # Flag extreme adult:child ratios (< 0.3 or > 5)
    df['extreme_ratio'] = (df['adult_child_ratio'] < 0.3) | (df['adult_child_ratio'] > 5)
    
    # Combined demographic anomaly flag
    df['demographic_anomaly'] = df['high_imbalance'] | df['extreme_ratio']
    
    return df


def get_demographic_summary(df: pd.DataFrame, group_cols: Optional[list] = None) -> pd.DataFrame:
    """
    Generate summary of demographic patterns.
    
    Args:
        df: DataFrame with demographic data
        group_cols: Columns to group by (e.g., ['state'])
        
    Returns:
        Summary DataFrame
    """
    if 'total_enrolments' not in df.columns:
        df['total_enrolments'] = df['age_0_5'] + df['age_5_17'] + df['age_18_greater']
    
    if group_cols:
        summary = df.groupby(group_cols).agg({
            'age_0_5': 'sum',
            'age_5_17': 'sum',
            'age_18_greater': 'sum',
            'total_enrolments': 'sum'
        }).reset_index()
    else:
        summary = pd.DataFrame({
            'age_0_5': [df['age_0_5'].sum()],
            'age_5_17': [df['age_5_17'].sum()],
            'age_18_greater': [df['age_18_greater'].sum()],
            'total_enrolments': [df['total_enrolments'].sum()]
        })
    
    # Calculate percentages
    summary['pct_0_5'] = (summary['age_0_5'] / summary['total_enrolments'] * 100).round(2)
    summary['pct_5_17'] = (summary['age_5_17'] / summary['total_enrolments'] * 100).round(2)
    summary['pct_18_greater'] = (summary['age_18_greater'] / summary['total_enrolments'] * 100).round(2)
    
    return summary


if __name__ == "__main__":
    # Test demographic features
    from src.data.loader import load_all_data
    from src.data.preprocessor import clean_data, aggregate_by_state
    
    print("Testing demographic features...")
    df = load_all_data()
    df = clean_data(df)
    state_df = aggregate_by_state(df)
    
    print("\nCalculating age imbalance...")
    state_df = calculate_age_imbalance(state_df)
    print(state_df[['state', 'total_enrolments', 'demographic_imbalance']].head())
    
    print("\nCalculating adult:child ratio...")
    state_df = calculate_adult_child_ratio(state_df)
    print(state_df[['state', 'adult_child_ratio', 'child_dominance']].head())
    
    print("\nDemographic summary by state:")
    summary = get_demographic_summary(df.assign(
        total_enrolments=lambda x: x['age_0_5'] + x['age_5_17'] + x['age_18_greater']
    ), group_cols=['state'])
    print(summary)
