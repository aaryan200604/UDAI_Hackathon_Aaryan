"""
Unit Tests for Anomaly Detection Module
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.anomaly_detector import AnomalyDetector, detect_anomalies, get_anomaly_summary


class TestAnomalyDetector:
    """Test AnomalyDetector class."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 100
        
        self.df = pd.DataFrame({
            'total_enrolments': np.random.normal(1000, 100, n_samples),
            'growth_rate': np.random.normal(0.05, 0.1, n_samples),
            'volatility_cv': np.random.exponential(0.3, n_samples),
            'demographic_imbalance': np.random.uniform(0, 0.2, n_samples)
        })
        
        # Add some anomalies
        self.df.loc[5, 'total_enrolments'] = 5000
        self.df.loc[15, 'growth_rate'] = 2.0
        self.df.loc[25, 'volatility_cv'] = 3.0
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        detector = AnomalyDetector(contamination=0.1)
        assert detector.contamination == 0.1
        assert detector.model is None
        assert detector.feature_names is None
    
    def test_fit_predict(self):
        """Test fit and predict."""
        detector = AnomalyDetector(contamination=0.1)
        feature_cols = ['total_enrolments', 'growth_rate', 'volatility_cv']
        
        result = detector.fit_predict(self.df, feature_cols)
        
        assert 'is_anomaly' in result.columns
        assert 'anomaly_score' in result.columns
        assert 'anomaly_score_normalized' in result.columns
        
        # Check that some anomalies were detected
        assert result['is_anomaly'].sum() > 0
        assert result['is_anomaly'].sum() < len(result)
    
    def test_separate_fit_and_predict(self):
        """Test separate fit and predict calls."""
        detector = AnomalyDetector()
        feature_cols = ['total_enrolments', 'growth_rate']
        
        # Fit
        detector.fit(self.df, feature_cols)
        assert detector.model is not None
        assert detector.feature_names == feature_cols
        
        # Predict
        result = detector.predict(self.df)
        assert 'is_anomaly' in result.columns
    
    def test_predict_without_fit_raises_error(self):
        """Test that predict without fit raises error."""
        detector = AnomalyDetector()
        
        with pytest.raises(RuntimeError):
            detector.predict(self.df)
    
    def test_anomaly_scores_range(self):
        """Test that normalized anomaly scores are in [0, 1]."""
        detector = AnomalyDetector()
        feature_cols = ['total_enrolments', 'growth_rate']
        
        result = detector.fit_predict(self.df, feature_cols)
        
        assert result['anomaly_score_normalized'].min() >= 0
        assert result['anomaly_score_normalized'].max() <= 1


class TestDetectAnomalies:
    """Test detect_anomalies helper function."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.df = pd.DataFrame({
            'state': ['State A'] * 50,
            'total_enrolments': np.random.normal(1000, 100, 50),
            'growth_rate': np.random.normal(0.05, 0.1, 50),
            'volatility_cv': np.random.exponential(0.3, 50)
        })
    
    def test_detect_anomalies_basic(self):
        """Test basic anomaly detection."""
        result = detect_anomalies(self.df)
        
        assert 'is_anomaly' in result.columns
        assert 'anomaly_score' in result.columns
    
    def test_detect_anomalies_with_custom_features(self):
        """Test with custom feature columns."""
        feature_cols = ['total_enrolments', 'growth_rate']
        result = detect_anomalies(self.df, feature_cols=feature_cols)
        
        assert 'is_anomaly' in result.columns
    
    def test_detect_anomalies_with_custom_contamination(self):
        """Test with custom contamination rate."""
        result = detect_anomalies(self.df, contamination=0.2)
        
        anomaly_rate = result['is_anomaly'].sum() / len(result)
        # Should be around 0.2
        assert 0.1 < anomaly_rate < 0.3
    
    def test_detect_anomalies_insufficient_features(self):
        """Test with insufficient features."""
        df = pd.DataFrame({
            'total_enrolments': [100, 200, 300]
        })
        
        result = detect_anomalies(df)
        
        # Should handle gracefully
        assert 'is_anomaly' in result.columns
        assert not result['is_anomaly'].any()


class TestGetAnomalySummary:
    """Test anomaly summary function."""
    
    def setup_method(self):
        """Set up test data."""
        self.df = pd.DataFrame({
            'state': ['State A', 'State B', 'State C'],
            'district': ['D1', 'D2', 'D3'],
            'is_anomaly': [True, False, True],
            'anomaly_score': [0.8, 0.2, 0.9]
        })
    
    def test_summary_structure(self):
        """Test summary structure."""
        summary = get_anomaly_summary(self.df)
        
        assert 'total_records' in summary
        assert 'anomaly_count' in summary
        assert 'anomaly_rate' in summary
        assert 'avg_anomaly_score' in summary
        assert 'top_anomalies' in summary
    
    def test_summary_values(self):
        """Test summary values."""
        summary = get_anomaly_summary(self.df)
        
        assert summary['total_records'] == 3
        assert summary['anomaly_count'] == 2
        assert summary['anomaly_rate'] == pytest.approx(66.67, abs=0.1)
    
    def test_summary_without_anomalies(self):
        """Test summary with no anomalies."""
        df = pd.DataFrame({
            'is_anomaly': [False, False, False],
            'anomaly_score': [0.1, 0.2, 0.15]
        })
        
        summary = get_anomaly_summary(df)
        
        assert summary['anomaly_count'] == 0
        assert len(summary['top_anomalies']) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
