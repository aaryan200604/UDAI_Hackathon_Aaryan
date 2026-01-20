from .temporal import calculate_growth_rate, calculate_volatility, calculate_seasonal_indicator
from .demographic import calculate_age_imbalance, calculate_adult_child_ratio
from ..models.statistical import zscore_analysis

import pandas as pd
from typing import List, Optional

class FeatureEngineer:
    """
    Unified Feature Engineering Layer.
    Orchestrates temporal, demographic, and statistical feature generation.
    """
    
    def __init__(self):
        pass
        
    def enrich_data(self, df: pd.DataFrame, 
                   group_cols: List[str] = ['state', 'district'],
                   value_col: str = 'enrolment_count') -> pd.DataFrame:
        """
        Apply all feature engineering steps to the dataframe.
        """
        df = df.copy()
        
        # 1. Temporal Indicators
        df = calculate_growth_rate(df, value_col=value_col, group_cols=group_cols)
        df = calculate_volatility(df, value_col=value_col, group_cols=group_cols)
        
        # Seasonal indicators (often monthly)
        if 'date' in df.columns:
            # We assume date column is already datetime
            df = calculate_seasonal_indicator(df, value_col=value_col)
            
        # 2. Demographic Indicators
        # Requires age columns to be present: age_0_5, age_5_17, age_18_greater
        age_cols = ['age_0_5', 'age_5_17', 'age_18_greater']
        if all(col in df.columns for col in age_cols):
             df = calculate_age_imbalance(df)
             df = calculate_adult_child_ratio(df)
             
        # 3. Statistical / Spikes
        # We can treat Z-score > 3 as a 'spike' feature
        df = zscore_analysis(df, value_col=value_col, group_cols=group_cols)
        df.rename(columns={'zscore': 'enrolment_zscore', 'zscore_outlier': 'is_spike'}, inplace=True)
        
        return df
