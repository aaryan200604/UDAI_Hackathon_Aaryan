"""
Statistical Analysis Module
Implements Z-Score and IQR-based outlier detection for validation.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional, List, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import ZSCORE_THRESHOLD


def zscore_analysis(df: pd.DataFrame,
                    value_col: str,
                    threshold: float = None,
                    group_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Perform Z-score based outlier detection.
    
    Args:
        df: DataFrame with values
        value_col: Column to analyze
        threshold: Z-score threshold for outliers (default from config)
        group_cols: Optional grouping columns
        
    Returns:
        DataFrame with Z-scores and outlier flags
    """
    if threshold is None:
        threshold = ZSCORE_THRESHOLD
    
    df = df.copy()
    
    if group_cols:
        # Calculate Z-score within groups
        df['zscore'] = df.groupby(group_cols)[value_col].transform(
            lambda x: stats.zscore(x, nan_policy='omit')
        )
    else:
        # Calculate overall Z-score
        df['zscore'] = stats.zscore(df[value_col], nan_policy='omit')
    
    # Flag outliers
    df['zscore_outlier'] = df['zscore'].abs() > threshold
    
    # Outlier direction
    df['zscore_direction'] = np.where(df['zscore'] > threshold, 'high',
                                      np.where(df['zscore'] < -threshold, 'low', 'normal'))
    
    return df


def iqr_analysis(df: pd.DataFrame,
                 value_col: str,
                 multiplier: float = 1.5,
                 group_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Perform IQR-based outlier detection.
    
    Args:
        df: DataFrame with values
        value_col: Column to analyze
        multiplier: IQR multiplier for bounds (1.5 = mild, 3 = extreme)
        group_cols: Optional grouping columns
        
    Returns:
        DataFrame with IQR bounds and outlier flags
    """
    df = df.copy()
    
    def calculate_iqr_bounds(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
        return lower, upper
    
    if group_cols:
        # Calculate IQR within groups
        bounds = df.groupby(group_cols)[value_col].apply(
            lambda x: pd.Series(calculate_iqr_bounds(x), index=['iqr_lower', 'iqr_upper'])
        ).unstack()
        
        df = df.merge(bounds, on=group_cols, how='left')
    else:
        lower, upper = calculate_iqr_bounds(df[value_col])
        df['iqr_lower'] = lower
        df['iqr_upper'] = upper
    
    # Flag outliers
    df['iqr_outlier'] = (df[value_col] < df['iqr_lower']) | (df[value_col] > df['iqr_upper'])
    
    # Outlier direction
    df['iqr_direction'] = np.where(df[value_col] > df['iqr_upper'], 'high',
                                   np.where(df[value_col] < df['iqr_lower'], 'low', 'normal'))
    
    return df


def detect_statistical_outliers(df: pd.DataFrame,
                                 value_cols: List[str],
                                 method: str = 'both',
                                 zscore_threshold: float = None,
                                 iqr_multiplier: float = 1.5) -> pd.DataFrame:
    """
    Detect statistical outliers using Z-score and/or IQR methods.
    
    Args:
        df: DataFrame with values
        value_cols: List of columns to analyze
        method: Detection method ('zscore', 'iqr', or 'both')
        zscore_threshold: Z-score threshold
        iqr_multiplier: IQR multiplier
        
    Returns:
        DataFrame with outlier flags for each column
    """
    df = df.copy()
    
    for col in value_cols:
        if col not in df.columns:
            print(f"Warning: Column {col} not found in DataFrame")
            continue
        
        if method in ['zscore', 'both']:
            df = zscore_analysis(df, col, threshold=zscore_threshold)
            df = df.rename(columns={
                'zscore': f'{col}_zscore',
                'zscore_outlier': f'{col}_zscore_outlier',
                'zscore_direction': f'{col}_zscore_direction'
            })
        
        if method in ['iqr', 'both']:
            df = iqr_analysis(df, col, multiplier=iqr_multiplier)
            df = df.rename(columns={
                'iqr_lower': f'{col}_iqr_lower',
                'iqr_upper': f'{col}_iqr_upper',
                'iqr_outlier': f'{col}_iqr_outlier',
                'iqr_direction': f'{col}_iqr_direction'
            })
    
    # Create combined outlier flag
    if method == 'both':
        outlier_cols = [f'{col}_zscore_outlier' for col in value_cols if f'{col}_zscore_outlier' in df.columns]
        outlier_cols += [f'{col}_iqr_outlier' for col in value_cols if f'{col}_iqr_outlier' in df.columns]
        df['any_outlier'] = df[outlier_cols].any(axis=1) if outlier_cols else False
        df['outlier_count'] = df[outlier_cols].sum(axis=1) if outlier_cols else 0
    
    return df


def calculate_percentile_rank(df: pd.DataFrame,
                              value_col: str,
                              group_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calculate percentile rank for values.
    
    Args:
        df: DataFrame with values
        value_col: Column to rank
        group_cols: Optional grouping columns
        
    Returns:
        DataFrame with percentile ranks
    """
    df = df.copy()
    
    if group_cols:
        df['percentile_rank'] = df.groupby(group_cols)[value_col].transform(
            lambda x: x.rank(pct=True) * 100
        )
    else:
        df['percentile_rank'] = df[value_col].rank(pct=True) * 100
    
    return df


def get_distribution_stats(df: pd.DataFrame, value_col: str) -> dict:
    """
    Calculate comprehensive distribution statistics.
    
    Args:
        df: DataFrame with values
        value_col: Column to analyze
        
    Returns:
        Dictionary with distribution statistics
    """
    values = df[value_col].dropna()
    
    stats_dict = {
        'count': len(values),
        'mean': values.mean(),
        'std': values.std(),
        'min': values.min(),
        'max': values.max(),
        'median': values.median(),
        'q1': values.quantile(0.25),
        'q3': values.quantile(0.75),
        'iqr': values.quantile(0.75) - values.quantile(0.25),
        'skewness': values.skew(),
        'kurtosis': values.kurtosis()
    }
    
    # Normality test
    if len(values) >= 8:
        _, p_value = stats.shapiro(values.sample(min(5000, len(values))))
        stats_dict['normality_pvalue'] = p_value
        stats_dict['is_normal'] = p_value > 0.05
    
    return stats_dict


if __name__ == "__main__":
    # Test statistical analysis
    print("Testing statistical analysis module...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    
    sample_data = pd.DataFrame({
        'state': ['State A'] * 50 + ['State B'] * 50,
        'value': np.concatenate([
            np.random.normal(100, 15, 45),  # Normal values
            [200, 250, 20, 15, 300],  # Outliers
            np.random.normal(150, 20, 45),  # Normal values
            [400, 450, 30, 25, 500]  # Outliers
        ])
    })
    
    print("\nZ-score analysis:")
    df_z = zscore_analysis(sample_data, 'value', threshold=2)
    print(f"  Outliers detected: {df_z['zscore_outlier'].sum()}")
    print(df_z[df_z['zscore_outlier']][['value', 'zscore', 'zscore_direction']])
    
    print("\nIQR analysis:")
    df_iqr = iqr_analysis(sample_data, 'value')
    print(f"  Outliers detected: {df_iqr['iqr_outlier'].sum()}")
    print(df_iqr[df_iqr['iqr_outlier']][['value', 'iqr_lower', 'iqr_upper', 'iqr_direction']])
    
    print("\nDistribution stats:")
    stats_dict = get_distribution_stats(sample_data, 'value')
    for key, value in stats_dict.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
