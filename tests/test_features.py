"""
Unit Tests for Feature Engineering Module
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.temporal import calculate_growth_rate, calculate_volatility
from src.features.demographic import calculate_age_imbalance, calculate_adult_child_ratio
from src.features.risk_scoring import calculate_risk_score, classify_risk_level, normalize_feature


class TestTemporalFeatures:
    """Test temporal feature engineering functions."""
    
    def setup_method(self):
        """Set up test data."""
        dates = pd.date_range('2025-01-01', periods=10)
        self.df = pd.DataFrame({
            'date': dates,
            'state': ['State A'] * 10,
            'total_enrolments': [100, 110, 105, 115, 120, 125, 130, 128, 135, 140]
        })
    
    def test_growth_rate_calculation(self):
        """Test growth rate calculation."""
        result = calculate_growth_rate(self.df)
        
        assert 'growth_rate' in result.columns
        assert result['growth_rate'].notna().sum() == 9  # First row is NaN
        assert result['growth_rate'].iloc[1] == pytest.approx(0.1, abs=0.01)
    
    def test_growth_rate_with_grouping(self):
        """Test growth rate with state grouping."""
        df_multi = pd.concat([
            self.df,
            self.df.assign(state='State B')
        ])
        
        result = calculate_growth_rate(df_multi, group_cols=['state'])
        
        assert 'growth_rate' in result.columns
        assert len(result) == 20
    
    def test_volatility_calculation(self):
        """Test volatility calculation."""
        result = calculate_volatility(self.df, window=3)
        
        assert 'volatility' in result.columns
        assert 'volatility_cv' in result.columns
        assert result['volatility'].notna().any()


class TestDemographicFeatures:
    """Test demographic feature engineering functions."""
    
    def setup_method(self):
        """Set up test data."""
        self.df = pd.DataFrame({
            'state': ['State A', 'State B', 'State C'],
            'age_0_5': [100, 150, 80],
            'age_5_17': [250, 300, 200],
            'age_18_greater': [650, 550, 720]
        })
    
    def test_age_imbalance_calculation(self):
        """Test age imbalance calculation."""
        result = calculate_age_imbalance(self.df)
        
        assert 'demographic_imbalance' in result.columns
        assert 'ratio_0_5' in result.columns
        assert 'ratio_5_17' in result.columns
        assert 'ratio_18_greater' in result.columns
        
        # Check that ratios sum to 1
        ratio_sum = result['ratio_0_5'] + result['ratio_5_17'] + result['ratio_18_greater']
        assert all(ratio_sum.round(2) == 1.0)
    
    def test_adult_child_ratio_calculation(self):
        """Test adult:child ratio calculation."""
        result = calculate_adult_child_ratio(self.df)
        
        assert 'adult_child_ratio' in result.columns
        assert 'total_children' in result.columns
        assert 'child_dominance' in result.columns
        
        # Check that ratios are positive
        assert all(result['adult_child_ratio'] > 0)
    
    def test_demographic_imbalance_with_custom_ratios(self):
        """Test with custom expected ratios."""
        custom_ratios = {
            'age_0_5': 0.15,
            'age_5_17': 0.30,
            'age_18_greater': 0.55
        }
        
        result = calculate_age_imbalance(self.df, expected_ratios=custom_ratios)
        assert 'demographic_imbalance' in result.columns


class TestRiskScoring:
    """Test risk scoring functions."""
    
    def setup_method(self):
        """Set up test data."""
        self.df = pd.DataFrame({
            'state': ['State A', 'State B', 'State C', 'State D'],
            'volatility_cv': [0.1, 0.5, 0.3, 0.8],
            'demographic_imbalance': [0.05, 0.15, 0.10, 0.25],
            'growth_rate': [0.02, 0.40, 0.10, 0.75],
            'seasonal_deviation': [0.05, 0.20, 0.10, 0.45]
        })
    
    def test_normalize_feature(self):
        """Test feature normalization."""
        series = pd.Series([0, 25, 50, 75, 100])
        
        # Min-max normalization
        normalized = normalize_feature(series, method='minmax')
        assert normalized.min() == 0.0
        assert normalized.max() == 1.0
        assert normalized.iloc[2] == pytest.approx(0.5, abs=0.01)
    
    def test_risk_score_calculation(self):
        """Test risk score calculation."""
        result = calculate_risk_score(self.df)
        
        assert 'risk_score' in result.columns
        assert all(result['risk_score'] >= 0)
        assert all(result['risk_score'] <= 1)
        
        # State D should have highest risk (all metrics highest)
        assert result.loc[3, 'risk_score'] == result['risk_score'].max()
    
    def test_risk_classification(self):
        """Test risk level classification."""
        df_with_scores = pd.DataFrame({
            'risk_score': [0.2, 0.4, 0.7, 0.9]
        })
        
        result = classify_risk_level(df_with_scores)
        
        assert 'risk_level' in result.columns
        assert 'risk_tier' in result.columns
        
        assert result['risk_level'].iloc[0] == 'Low'
        assert result['risk_level'].iloc[1] == 'Medium'
        assert result['risk_level'].iloc[2] == 'High'
        assert result['risk_level'].iloc[3] == 'Critical'
    
    def test_risk_score_with_missing_features(self):
        """Test risk scoring with missing features."""
        incomplete_df = pd.DataFrame({
            'state': ['State A'],
            'volatility_cv': [0.5]
        })
        
        result = calculate_risk_score(incomplete_df)
        assert 'risk_score' in result.columns
        assert not result['risk_score'].isna().any()


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        
        # Should not raise errors
        try:
            result = calculate_risk_score(df)
            assert len(result) == 0
        except Exception:
            pytest.fail("Should handle empty DataFrame gracefully")
    
    def test_single_row(self):
        """Test with single row."""
        df = pd.DataFrame({
            'date': [datetime.now()],
            'state': ['State A'],
            'total_enrolments': [1000],
            'age_0_5': [100],
            'age_5_17': [300],
            'age_18_greater': [600]
        })
        
        result = calculate_age_imbalance(df)
        assert len(result) == 1
        assert 'demographic_imbalance' in result.columns
    
    def test_all_zeros(self):
        """Test with all zero values."""
        df = pd.DataFrame({
            'age_0_5': [0, 0, 0],
            'age_5_17': [0, 0, 0],
            'age_18_greater': [0, 0, 0]
        })
        
        result = calculate_age_imbalance(df)
        # Should handle division by zero gracefully - column should exist
        assert 'demographic_imbalance' in result.columns
        # Values may be NaN due to division by zero, which is acceptable
    
    def test_missing_values(self):
        """Test with missing values."""
        df = pd.DataFrame({
            'date': pd.date_range('2025-01-01', periods=5),
            'total_enrolments': [100, np.nan, 120, np.nan, 140]
        })
        
        result = calculate_growth_rate(df)
        assert 'growth_rate' in result.columns


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v'])
