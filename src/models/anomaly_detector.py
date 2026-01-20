"""
Anomaly Detection Module
Implements Isolation Forest and other ML-based anomaly detection for UIDAI data.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Optional, List, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import ISOLATION_FOREST_CONTAMINATION


class AnomalyDetector:
    """
    ML-based anomaly detector using Isolation Forest.
    """
    
    def __init__(self, contamination: float = None, random_state: int = 42):
        """
        Initialize the anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies
            random_state: Random seed for reproducibility
        """
        if contamination is None:
            contamination = ISOLATION_FOREST_CONTAMINATION
            
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def fit(self, df: pd.DataFrame, feature_cols: List[str]) -> 'AnomalyDetector':
        """
        Fit the anomaly detector on training data.
        
        Args:
            df: DataFrame with features
            feature_cols: List of feature columns to use
            
        Returns:
            Self for method chaining
        """
        self.feature_names = feature_cols
        
        # Prepare features
        X = df[feature_cols].copy()
        X = X.fillna(X.median())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Isolation Forest
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100,
            max_samples='auto'
        )
        self.model.fit(X_scaled)
        
        return self
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict anomalies in new data.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with anomaly predictions and scores
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        df = df.copy()
        
        # Prepare features
        X = df[self.feature_names].copy()
        X = X.fillna(X.median())
        X_scaled = self.scaler.transform(X)
        
        # Predict (-1 = anomaly, 1 = normal)
        predictions = self.model.predict(X_scaled)
        df['is_anomaly'] = predictions == -1
        
        # Get anomaly scores (lower = more anomalous)
        scores = self.model.decision_function(X_scaled)
        df['anomaly_score'] = -scores  # Invert so higher = more anomalous
        
        # Normalize score to 0-1
        df['anomaly_score_normalized'] = (
            (df['anomaly_score'] - df['anomaly_score'].min()) /
            (df['anomaly_score'].max() - df['anomaly_score'].min())
        )
        
        return df
    
    def fit_predict(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        Fit and predict in one step.
        
        Args:
            df: DataFrame with features
            feature_cols: List of feature columns to use
            
        Returns:
            DataFrame with anomaly predictions
        """
        self.fit(df, feature_cols)
        return self.predict(df)


def detect_anomalies(df: pd.DataFrame,
                     feature_cols: Optional[List[str]] = None,
                     contamination: float = None) -> pd.DataFrame:
    """
    Detect anomalies in enrolment data using Isolation Forest.
    
    Args:
        df: DataFrame with enrolment data and features
        feature_cols: List of feature columns to use for detection
        contamination: Expected anomaly rate
        
    Returns:
        DataFrame with anomaly flags and scores
    """
    if feature_cols is None:
        # Default feature columns
        default_features = [
            'total_enrolments',
            'growth_rate',
            'volatility_cv',
            'demographic_imbalance',
            'adult_child_ratio'
        ]
        feature_cols = [col for col in default_features if col in df.columns]
    
    if len(feature_cols) < 2:
        print(f"Warning: Only {len(feature_cols)} features available. Need at least 2.")
        df['is_anomaly'] = False
        df['anomaly_score'] = 0
        return df
    
    detector = AnomalyDetector(contamination=contamination)
    return detector.fit_predict(df, feature_cols)


def get_anomaly_summary(df: pd.DataFrame) -> dict:
    """
    Generate summary of detected anomalies.
    
    Args:
        df: DataFrame with anomaly predictions
        
    Returns:
        Summary dictionary
    """
    if 'is_anomaly' not in df.columns:
        return {'error': 'No anomaly detection results found'}
    
    anomalies = df[df['is_anomaly']]
    
    summary = {
        'total_records': len(df),
        'anomaly_count': len(anomalies),
        'anomaly_rate': len(anomalies) / len(df) * 100,
        'avg_anomaly_score': anomalies['anomaly_score'].mean() if len(anomalies) > 0 else 0,
        'top_anomalies': anomalies.nlargest(10, 'anomaly_score')[
            ['state', 'district', 'anomaly_score'] if 'district' in df.columns 
            else ['state', 'anomaly_score']
        ].to_dict('records') if len(anomalies) > 0 else []
    }
    
    return summary


def get_feature_importance(detector: AnomalyDetector, df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate feature importance for anomaly detection.
    
    Args:
        detector: Fitted AnomalyDetector
        df: DataFrame used for detection
        
    Returns:
        DataFrame with feature importance scores
    """
    if detector.model is None:
        raise RuntimeError("Model not fitted")
    
    # Use feature importances from Isolation Forest
    # (based on average path length contribution)
    importances = np.zeros(len(detector.feature_names))
    
    for tree in detector.model.estimators_:
        importances += tree.feature_importances_
    
    importances /= len(detector.model.estimators_)
    
    importance_df = pd.DataFrame({
        'feature': detector.feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return importance_df


if __name__ == "__main__":
    # Test anomaly detection
    print("Testing anomaly detector...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    
    sample_data = pd.DataFrame({
        'state': ['State A'] * n_samples,
        'district': [f'District {i}' for i in range(n_samples)],
        'total_enrolments': np.random.normal(1000, 200, n_samples),
        'growth_rate': np.random.normal(0.05, 0.1, n_samples),
        'volatility_cv': np.random.exponential(0.3, n_samples),
        'demographic_imbalance': np.random.uniform(0, 0.3, n_samples)
    })
    
    # Add some anomalies
    sample_data.loc[5, 'total_enrolments'] = 5000  # Spike
    sample_data.loc[15, 'growth_rate'] = 2.0  # Extreme growth
    sample_data.loc[25, 'volatility_cv'] = 3.0  # High volatility
    
    print("\nDetecting anomalies...")
    result = detect_anomalies(sample_data, contamination=0.1)
    
    print("\nAnomaly Summary:")
    summary = get_anomaly_summary(result)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\nDetected Anomalies:")
    print(result[result['is_anomaly']][['district', 'total_enrolments', 'anomaly_score']])
