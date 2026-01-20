"""
UIDAI Aadhaar Data Vault - Main Entry Point
ML-Driven Anomaly Detection & Risk Intelligence for Aadhaar Enrolment Data

UIDAI Data Hackathon 2026
"""

import argparse
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data.loader import load_all_data, get_data_summary
from src.data.preprocessor import clean_data, validate_data, aggregate_by_state, aggregate_by_district
from src.features.temporal import calculate_growth_rate, calculate_volatility, detect_trends
from src.features.demographic import calculate_age_imbalance, calculate_adult_child_ratio, get_demographic_summary
from src.features.risk_scoring import calculate_risk_score, classify_risk_level
from src.models.anomaly_detector import detect_anomalies, get_anomaly_summary
from src.models.statistical import detect_statistical_outliers, get_distribution_stats


def run_full_analysis(output_dir: str = None):
    """
    Run the complete analysis pipeline.
    
    Args:
        output_dir: Directory to save output files
    """
    print("=" * 60)
    print("UIDAI Aadhaar Data Vault - Full Analysis")
    print("=" * 60)
    
    # 1. Load Data
    print("\n[1/6] Loading data...")
    df = load_all_data()
    
    # 2. Clean and Validate
    print("\n[2/6] Cleaning and validating data...")
    df = clean_data(df)
    is_valid, report = validate_data(df)
    print(f"Validation: {'PASSED' if is_valid else 'FAILED'}")
    if report['warnings']:
        print(f"Warnings: {report['warnings']}")
    
    # 3. Aggregate Data
    print("\n[3/6] Aggregating data...")
    state_df = aggregate_by_state(df)
    district_df = aggregate_by_district(df)
    print(f"  State aggregation: {len(state_df)} records")
    print(f"  District aggregation: {len(district_df)} records")
    
    # 4. Feature Engineering
    print("\n[4/6] Engineering features...")
    state_df = calculate_growth_rate(state_df, group_cols=['state'])
    state_df = calculate_volatility(state_df, group_cols=['state'])
    state_df = detect_trends(state_df, group_cols=['state'])
    state_df = calculate_age_imbalance(state_df)
    state_df = calculate_adult_child_ratio(state_df)
    print("  [OK] Growth rate")
    print("  [OK] Volatility")
    print("  [OK] Trends")
    print("  [OK] Demographic indicators")
    
    # 5. Anomaly Detection
    print("\n[5/6] Detecting anomalies...")
    feature_cols = ['total_enrolments', 'growth_rate', 'volatility_cv', 'demographic_imbalance']
    feature_cols = [col for col in feature_cols if col in state_df.columns]
    state_df = detect_anomalies(state_df, feature_cols)
    
    anomaly_summary = get_anomaly_summary(state_df)
    print(f"  Total records analyzed: {anomaly_summary['total_records']}")
    print(f"  Anomalies detected: {anomaly_summary['anomaly_count']}")
    print(f"  Anomaly rate: {anomaly_summary['anomaly_rate']:.2f}%")
    
    # 6. Risk Scoring
    print("\n[6/6] Calculating risk scores...")
    state_df = calculate_risk_score(state_df)
    state_df = classify_risk_level(state_df)
    
    risk_summary = state_df.groupby('risk_level').size()
    print("  Risk Distribution:")
    for level, count in risk_summary.items():
        print(f"    {level}: {count}")
    
    # Summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    # Data Summary
    summary = get_data_summary(df)
    print(f"\nData Summary:")
    print(f"  Total records: {summary['total_records']:,}")
    print(f"  Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    print(f"  States: {summary['states']}")
    print(f"  Districts: {summary['districts']}")
    print(f"  Total enrolments: {summary['total_enrolments']['total']:,.0f}")
    
    # Demographic Summary
    print(f"\nAge Distribution:")
    total = summary['total_enrolments']['total']
    print(f"  Age 0-5: {summary['total_enrolments']['age_0_5']:,.0f} ({summary['total_enrolments']['age_0_5']/total*100:.1f}%)")
    print(f"  Age 5-17: {summary['total_enrolments']['age_5_17']:,.0f} ({summary['total_enrolments']['age_5_17']/total*100:.1f}%)")
    print(f"  Age 18+: {summary['total_enrolments']['age_18_greater']:,.0f} ({summary['total_enrolments']['age_18_greater']/total*100:.1f}%)")
    
    # Top Anomalies
    if anomaly_summary['top_anomalies']:
        print(f"\nTop Anomalies:")
        for i, anomaly in enumerate(anomaly_summary['top_anomalies'][:5], 1):
            print(f"  {i}. {anomaly.get('state', 'N/A')} - {anomaly.get('district', 'N/A')}: Score {anomaly['anomaly_score']:.3f}")
    
    # Save results if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        state_df.to_csv(os.path.join(output_dir, 'state_analysis.csv'), index=False)
        district_df.to_csv(os.path.join(output_dir, 'district_analysis.csv'), index=False)
        print(f"\nResults saved to: {output_dir}")
    
    return state_df, district_df


def run_dashboard():
    """Launch the Streamlit dashboard."""
    import subprocess
    app_path = os.path.join(PROJECT_ROOT, 'app', 'app.py')
    
    print("Launching UIDAI Data Vault Dashboard...")
    print("Open your browser at http://localhost:8501")
    
    subprocess.run(['streamlit', 'run', app_path])


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='UIDAI Aadhaar Data Vault - ML-Driven Analytics Platform'
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=['analysis', 'dashboard', 'both'],
        default='both',
        help='Run mode: analysis (CLI), dashboard (Streamlit), or both'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory for analysis results'
    )
    
    args = parser.parse_args()
    
    if args.mode in ['analysis', 'both']:
        run_full_analysis(args.output)
    
    if args.mode in ['dashboard', 'both']:
        run_dashboard()


if __name__ == "__main__":
    main()
