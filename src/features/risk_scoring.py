"""
Risk Scoring Module
Calculates composite risk scores and classifications for UIDAI governance.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import RISK_WEIGHTS


def normalize_feature(series: pd.Series, method: str = 'minmax') -> pd.Series:
    """
    Normalize a feature to 0-1 range.
    
    Args:
        series: Pandas Series to normalize
        method: Normalization method ('minmax' or 'zscore')
        
    Returns:
        Normalized Series
    """
    if method == 'minmax':
        min_val = series.min()
        max_val = series.max()
        if max_val - min_val == 0:
            return pd.Series([0.5] * len(series), index=series.index)
        return (series - min_val) / (max_val - min_val)
    elif method == 'zscore':
        mean_val = series.mean()
        std_val = series.std()
        if std_val == 0:
            return pd.Series([0] * len(series), index=series.index)
        return (series - mean_val) / std_val
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def calculate_risk_score(df: pd.DataFrame,
                         weights: Optional[Dict[str, float]] = None,
                         feature_cols: Optional[list] = None) -> pd.DataFrame:
    """
    Calculate composite risk score based on multiple indicators.
    
    Args:
        df: DataFrame with risk-related features
        weights: Weights for each feature (default from config)
        feature_cols: List of feature columns to use
        
    Returns:
        DataFrame with risk score added
    """
    if weights is None:
        weights = RISK_WEIGHTS
    
    df = df.copy()
    
    # Default feature columns if not specified
    if feature_cols is None:
        feature_cols = [
            'volatility_cv',           # Volatility score
            'demographic_imbalance',   # Demographic deviation
            'growth_rate',             # Unusual growth
            'seasonal_deviation'       # Seasonal anomaly
        ]
    
    # Only use features that exist in the dataframe
    available_features = [col for col in feature_cols if col in df.columns]
    
    if not available_features:
        print("Warning: No risk features found in dataframe")
        df['risk_score'] = 0.5
        return df
    
    # Normalize each feature
    normalized_features = {}
    for col in available_features:
        # Handle NaN values
        values = df[col].fillna(df[col].median())
        # For features where higher = more risk, normalize directly
        normalized_features[f'{col}_norm'] = normalize_feature(values.abs())
    
    # Calculate weighted sum
    df['risk_score'] = 0
    total_weight = 0
    
    for col in available_features:
        # Get weight for this feature
        weight = weights.get(col.replace('_norm', ''), 0.2)
        df['risk_score'] += normalized_features[f'{col}_norm'] * weight
        total_weight += weight
    
    # Normalize by total weight
    if total_weight > 0:
        df['risk_score'] = df['risk_score'] / total_weight
    
    # Ensure score is between 0 and 1
    df['risk_score'] = df['risk_score'].clip(0, 1)
    
    return df


def classify_risk_level(df: pd.DataFrame,
                        score_col: str = 'risk_score',
                        thresholds: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    Classify risk level based on composite score.
    
    Args:
        df: DataFrame with risk score
        score_col: Column name for risk score
        thresholds: Thresholds for risk levels
        
    Returns:
        DataFrame with risk level classification
    """
    if thresholds is None:
        thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }
    
    df = df.copy()
    
    conditions = [
        df[score_col] <= thresholds['low'],
        (df[score_col] > thresholds['low']) & (df[score_col] <= thresholds['medium']),
        (df[score_col] > thresholds['medium']) & (df[score_col] <= thresholds['high']),
        df[score_col] > thresholds['high']
    ]
    
    choices = ['Low', 'Medium', 'High', 'Critical']
    
    df['risk_level'] = np.select(conditions, choices, default='Unknown')
    
    # Numeric risk tier (1-4)
    df['risk_tier'] = np.select(conditions, [1, 2, 3, 4], default=0)
    
    return df


def generate_risk_explanation(row: pd.Series,
                              feature_cols: Optional[list] = None) -> str:
    """
    Generate human-readable explanation for risk score.
    
    Args:
        row: DataFrame row with features
        feature_cols: List of feature columns to explain
        
    Returns:
        Explanation string
    """
    if feature_cols is None:
        feature_cols = ['volatility_cv', 'demographic_imbalance', 'growth_rate']
    
    explanations = []
    
    # Check volatility
    if 'volatility_cv' in row.index and pd.notna(row['volatility_cv']):
        if row['volatility_cv'] > 0.5:
            explanations.append(f"High volatility (CV: {row['volatility_cv']:.2f})")
    
    # Check demographic imbalance
    if 'demographic_imbalance' in row.index and pd.notna(row['demographic_imbalance']):
        if row['demographic_imbalance'] > 0.15:
            explanations.append(f"Demographic imbalance detected ({row['demographic_imbalance']:.2f})")
    
    # Check growth rate
    if 'growth_rate' in row.index and pd.notna(row['growth_rate']):
        if abs(row['growth_rate']) > 0.5:
            direction = "surge" if row['growth_rate'] > 0 else "drop"
            explanations.append(f"Unusual enrolment {direction} ({row['growth_rate']*100:.1f}%)")
    
    # Check seasonal deviation
    if 'seasonal_deviation' in row.index and pd.notna(row['seasonal_deviation']):
        if abs(row['seasonal_deviation']) > 0.3:
            explanations.append(f"Seasonal pattern deviation ({row['seasonal_deviation']*100:.1f}%)")
    
    if not explanations:
        return "No significant risk factors identified"
    
    return "; ".join(explanations)


def get_high_risk_regions(df: pd.DataFrame,
                          risk_col: str = 'risk_score',
                          top_n: int = 10,
                          group_col: str = 'district') -> pd.DataFrame:
    """
    Get top high-risk regions/districts.
    
    Args:
        df: DataFrame with risk scores
        risk_col: Column for risk score
        top_n: Number of top regions to return
        group_col: Column to group by
        
    Returns:
        DataFrame with top high-risk regions
    """
    # Aggregate risk by region
    agg_df = df.groupby(group_col).agg({
        risk_col: 'mean',
        'total_enrolments': 'sum' if 'total_enrolments' in df.columns else 'count'
    }).reset_index()
    
    # Sort by risk score and get top N
    top_risk = agg_df.nlargest(top_n, risk_col)
    
    return top_risk


def calculate_district_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate comprehensive district-level risk score.
    
    Args:
        df: DataFrame with district-level data and features
        
    Returns:
        DataFrame with district risk scores
    """
    df = df.copy()
    
    # Aggregate by district if needed
    if 'date' in df.columns:
        district_df = df.groupby(['state', 'district']).agg({
            'total_enrolments': 'sum',
            'age_0_5': 'sum',
            'age_5_17': 'sum',
            'age_18_greater': 'sum',
            'volatility_cv': 'mean' if 'volatility_cv' in df.columns else 'first',
            'growth_rate': 'mean' if 'growth_rate' in df.columns else 'first',
            'demographic_imbalance': 'mean' if 'demographic_imbalance' in df.columns else 'first'
        }).reset_index()
    else:
        district_df = df
    
    # Calculate risk score
    district_df = calculate_risk_score(district_df)
    district_df = classify_risk_level(district_df)
    
    # Add explanation
    district_df['risk_explanation'] = district_df.apply(generate_risk_explanation, axis=1)
    
    return district_df


if __name__ == "__main__":
    # Test risk scoring
    print("Testing risk scoring module...")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'district': ['District A', 'District B', 'District C', 'District D'],
        'total_enrolments': [1000, 2000, 500, 1500],
        'volatility_cv': [0.1, 0.8, 0.3, 0.6],
        'demographic_imbalance': [0.05, 0.25, 0.10, 0.18],
        'growth_rate': [0.02, 0.75, -0.1, 0.4],
        'seasonal_deviation': [0.05, 0.45, 0.1, 0.2]
    })
    
    print("\nCalculating risk scores...")
    sample_data = calculate_risk_score(sample_data)
    sample_data = classify_risk_level(sample_data)
    
    print("\nRisk Assessment Results:")
    print(sample_data[['district', 'risk_score', 'risk_level']])
    
    print("\nRisk Explanations:")
    for _, row in sample_data.iterrows():
        print(f"  {row['district']}: {generate_risk_explanation(row)}")
