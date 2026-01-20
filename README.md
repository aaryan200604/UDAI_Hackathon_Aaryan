# UIDAI Aadhaar Data Vault

## ML-Driven Anomaly Detection & Risk Intelligence Platform

**UIDAI Data Hackathon 2026**

---

## 📋 Overview

This project implements a privacy-preserving, ML-driven analytics platform for analyzing UIDAI Aadhaar enrolment data. It provides:

- **Anomaly Detection**: Isolation Forest-based detection of unusual enrolment patterns
- **Risk Scoring**: Composite risk scores with explainable alerts
- **Trend Analysis**: Temporal patterns and seasonal indicators
- **Demographic Insights**: Age distribution analysis and imbalance detection
- **Interactive Dashboard**: Streamlit-based visualization platform

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Analysis

```bash
# Full analysis (CLI + Dashboard)
python main.py

# CLI analysis only
python main.py --mode analysis --output reports/

# Dashboard only
python main.py --mode dashboard
```

### 3. Access Dashboard

Open your browser at `http://localhost:8501`

---

## 📁 Project Structure

```
uidai_hackathon/
├── config.py                 # Configuration settings
├── main.py                   # Main entry point
├── requirements.txt          # Python dependencies
├── src/
│   ├── data/
│   │   ├── loader.py         # Data loading utilities
│   │   └── preprocessor.py   # Data cleaning & aggregation
│   ├── features/
│   │   ├── temporal.py       # Time-based features
│   │   ├── demographic.py    # Age-based features
│   │   └── risk_scoring.py   # Risk calculation
│   └── models/
│       ├── anomaly_detector.py  # Isolation Forest
│       └── statistical.py       # Z-Score/IQR analysis
├── app/
│   └── app.py                # Streamlit dashboard
├── notebooks/                # Jupyter analysis notebooks
└── reports/                  # Generated reports
```

---

## 🔧 Features

### Data Processing
- Load and merge multiple CSV files
- Clean and validate data
- Aggregate by state/district

### Feature Engineering
- **Temporal**: Growth rate, volatility, trend detection
- **Demographic**: Age imbalance, adult:child ratio
- **Risk**: Composite scoring with explainability

### ML Models
- **Isolation Forest**: Unsupervised anomaly detection
- **Z-Score/IQR**: Statistical outlier validation

### Visualization
- Interactive Streamlit dashboard
- Trend charts and heatmaps
- Risk distribution views
- Anomaly alerts

---

## 📊 Evaluation Criteria Alignment

| Criteria | Implementation |
|----------|----------------|
| Data Analysis & Insights | Comprehensive univariate/bivariate analysis |
| Creativity & Originality | ML + rule-based hybrid approach |
| Technical Implementation | Modular, documented code |
| Visualization | Interactive dashboard with Plotly |
| Impact & Applicability | Actionable risk intelligence |

---

## 👥 Team

UIDAI Data Hackathon 2026 Submission

---

## 📝 License

This project is created for the UIDAI Data Hackathon 2026.
