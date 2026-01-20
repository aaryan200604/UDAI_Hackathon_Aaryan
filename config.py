# UIDAI Data Hackathon Configuration

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')

# Original data location - making it relative for cloud compatibility
ORIGINAL_DATA_DIR = os.path.join(DATA_DIR, 'raw')

# Data files
DATA_FILES = [
    'api_data_aadhar_enrolment_0_500000.csv',
    'api_data_aadhar_enrolment_500000_1000000.csv',
    'api_data_aadhar_enrolment_1000000_1006029.csv'
]

# Feature engineering parameters
ROLLING_WINDOW = 7  # Days for rolling calculations
VOLATILITY_WINDOW = 14  # Days for volatility calculation

# Anomaly detection parameters
ISOLATION_FOREST_CONTAMINATION = 0.05  # Expected anomaly rate
ZSCORE_THRESHOLD = 3  # Z-score threshold for outliers

# Risk scoring weights
RISK_WEIGHTS = {
    'growth_rate_anomaly': 0.25,
    'volatility_score': 0.20,
    'demographic_imbalance': 0.25,
    'seasonal_deviation': 0.15,
    'statistical_outlier': 0.15
}

# Dashboard settings
DASHBOARD_HOST = '0.0.0.0'
DASHBOARD_PORT = 8501
