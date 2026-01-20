"""
Temporal Features Module
Calculates time-based indicators for enrolment trends.
"""

import pandas as pd
import numpy as np
from typing import Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import ROLLING_WINDOW, VOLATILITY_WINDOW


def calculate_growth_rate(df: pd.DataFrame, 
                          value_col: str = 'total_enrolments',
                          group_cols: Optional[list] = None,
                          periods: int = 1) -> pd.DataFrame:
    """
    Calculate period-over-period growth rate for enrolments.
    
    Args:
        df: DataFrame with enrolment data
        value_col: Column to calculate growth rate for
        group_cols: Columns to group by (e.g., ['state', 'district'])
        periods: Number of periods for growth calculation
        
    Returns:
        DataFrame with growth rate column added
    """
    df = df.copy().sort_values('date')
    
    if group_cols:
        df['growth_rate'] = df.groupby(group_cols)[value_col].pct_change(periods=periods)
    else:
        df['growth_rate'] = df[value_col].pct_change(periods=periods)
    
    # Handle infinite values from division by zero
    df['growth_rate'] = df['growth_rate'].replace([np.inf, -np.inf], np.nan)
    
    return df


def calculate_volatility(df: pd.DataFrame,
                         value_col: str = 'total_enrolments',
                         group_cols: Optional[list] = None,
                         window: int = None) -> pd.DataFrame:
    """
    Calculate rolling volatility (standard deviation) of enrolments.
    
    Args:
        df: DataFrame with enrolment data
        value_col: Column to calculate volatility for
        group_cols: Columns to group by
        window: Rolling window size (default from config)
        
    Returns:
        DataFrame with volatility column added
    """
    if window is None:
        window = VOLATILITY_WINDOW
    
    df = df.copy().sort_values('date')
    
    if group_cols:
        df['volatility'] = df.groupby(group_cols)[value_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
        # Normalize by mean for coefficient of variation
        df['volatility_cv'] = df.groupby(group_cols)[value_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).std() / 
                     x.rolling(window=window, min_periods=1).mean()
        )
    else:
        df['volatility'] = df[value_col].rolling(window=window, min_periods=1).std()
        df['volatility_cv'] = (
            df[value_col].rolling(window=window, min_periods=1).std() /
            df[value_col].rolling(window=window, min_periods=1).mean()
        )
    
    df['volatility_cv'] = df['volatility_cv'].replace([np.inf, -np.inf], np.nan)
    
    return df


def calculate_rolling_mean(df: pd.DataFrame,
                           value_col: str = 'total_enrolments',
                           group_cols: Optional[list] = None,
                           window: int = None) -> pd.DataFrame:
    """
    Calculate rolling mean for trend smoothing.
    
    Args:
        df: DataFrame with enrolment data
        value_col: Column to calculate rolling mean for
        group_cols: Columns to group by
        window: Rolling window size
        
    Returns:
        DataFrame with rolling mean column added
    """
    if window is None:
        window = ROLLING_WINDOW
    
    df = df.copy().sort_values('date')
    
    if group_cols:
        df['rolling_mean'] = df.groupby(group_cols)[value_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
    else:
        df['rolling_mean'] = df[value_col].rolling(window=window, min_periods=1).mean()
    
    return df


def detect_trends(df: pd.DataFrame,
                  value_col: str = 'total_enrolments',
                  group_cols: Optional[list] = None,
                  window: int = None) -> pd.DataFrame:
    """
    Detect upward/downward trends in enrolment data.
    
    Args:
        df: DataFrame with enrolment data
        value_col: Column to analyze for trends
        group_cols: Columns to group by
        window: Window size for trend detection
        
    Returns:
        DataFrame with trend indicators added
    """
    if window is None:
        window = ROLLING_WINDOW
    
    df = df.copy().sort_values('date')
    
    # Calculate short-term and long-term moving averages
    if group_cols:
        df['ma_short'] = df.groupby(group_cols)[value_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        df['ma_long'] = df.groupby(group_cols)[value_col].transform(
            lambda x: x.rolling(window=window*2, min_periods=1).mean()
        )
    else:
        df['ma_short'] = df[value_col].rolling(window=window, min_periods=1).mean()
        df['ma_long'] = df[value_col].rolling(window=window*2, min_periods=1).mean()
    
    # Trend direction: positive = upward, negative = downward
    df['trend_strength'] = (df['ma_short'] - df['ma_long']) / df['ma_long'].replace(0, np.nan)
    df['trend_direction'] = np.where(df['trend_strength'] > 0.05, 'upward',
                                     np.where(df['trend_strength'] < -0.05, 'downward', 'stable'))
    
    return df


def calculate_seasonal_indicator(df: pd.DataFrame,
                                 date_col: str = 'date',
                                 value_col: str = 'total_enrolments') -> pd.DataFrame:
    """
    Calculate seasonal indicators based on monthly patterns.
    
    Args:
        df: DataFrame with enrolment data
        date_col: Date column name
        value_col: Value column to analyze
        
    Returns:
        DataFrame with seasonal indicators added
    """
    df = df.copy()
    
    # Extract date components
    df['month'] = df[date_col].dt.month
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Calculate monthly average
    monthly_avg = df.groupby('month')[value_col].mean()
    overall_avg = df[value_col].mean()
    
    # Seasonal factor: ratio of monthly avg to overall avg
    df['seasonal_factor'] = df['month'].map(monthly_avg) / overall_avg
    
    # Deviation from expected (seasonal) value
    df['seasonal_deviation'] = df[value_col] / df['month'].map(monthly_avg) - 1
    df['seasonal_deviation'] = df['seasonal_deviation'].replace([np.inf, -np.inf], np.nan)
    
    return df


if __name__ == "__main__":
    # Test temporal features
    from src.data.loader import load_all_data
    from src.data.preprocessor import clean_data, aggregate_by_state
    
    print("Testing temporal features...")
    df = load_all_data()
    df = clean_data(df)
    state_df = aggregate_by_state(df)
    
    print("\nCalculating growth rate...")
    state_df = calculate_growth_rate(state_df, group_cols=['state'])
    print(state_df[['date', 'state', 'total_enrolments', 'growth_rate']].head(20))
    
    print("\nCalculating volatility...")
    state_df = calculate_volatility(state_df, group_cols=['state'])
    print(state_df[['date', 'state', 'total_enrolments', 'volatility']].head(20))
