"""
Export & Reporting Module
Generates PDF and Excel reports for UIDAI enrolment analysis.
"""

import pandas as pd
import os
from typing import Optional, Dict, List
from datetime import datetime
import io


def export_to_excel(
    df: pd.DataFrame,
    output_path: str,
    sheet_name: str = 'Analysis',
    include_summary: bool = True
) -> str:
    """
    Export DataFrame to Excel with optional summary sheet.
    
    Args:
        df: DataFrame to export
        output_path: Path for output file
        sheet_name: Name of main data sheet
        include_summary: Whether to include a summary sheet
        
    Returns:
        Path to created file
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Main data sheet
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        if include_summary:
            # Create summary statistics
            summary_data = {
                'Metric': [
                    'Total Records',
                    'Date Range Start',
                    'Date Range End',
                    'Unique States',
                    'Unique Districts',
                    'Total Enrolments',
                    'Avg Risk Score',
                    'High Risk Records',
                    'Anomalies Detected'
                ],
                'Value': [
                    len(df),
                    df['date'].min() if 'date' in df.columns else 'N/A',
                    df['date'].max() if 'date' in df.columns else 'N/A',
                    df['state'].nunique() if 'state' in df.columns else 'N/A',
                    df['district'].nunique() if 'district' in df.columns else 'N/A',
                    df['total_enrolments'].sum() if 'total_enrolments' in df.columns else 'N/A',
                    f"{df['risk_score'].mean():.3f}" if 'risk_score' in df.columns else 'N/A',
                    len(df[df['risk_level'].isin(['High', 'Critical'])]) if 'risk_level' in df.columns else 'N/A',
                    df['is_anomaly'].sum() if 'is_anomaly' in df.columns else 'N/A'
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Risk distribution sheet
            if 'risk_level' in df.columns:
                risk_dist = df['risk_level'].value_counts().reset_index()
                risk_dist.columns = ['Risk Level', 'Count']
                risk_dist.to_excel(writer, sheet_name='Risk Distribution', index=False)
    
    return output_path


def export_anomalies_report(
    df: pd.DataFrame,
    output_path: str,
    top_n: int = 100
) -> str:
    """
    Export a focused report on detected anomalies.
    
    Args:
        df: DataFrame with anomaly detection results
        output_path: Path for output file
        top_n: Number of top anomalies to include
        
    Returns:
        Path to created file
    """
    if 'is_anomaly' not in df.columns:
        raise ValueError("DataFrame must contain anomaly detection results")
    
    anomalies = df[df['is_anomaly']].copy()
    
    if 'anomaly_score' in anomalies.columns:
        anomalies = anomalies.nlargest(top_n, 'anomaly_score')
    else:
        anomalies = anomalies.head(top_n)
    
    # Select relevant columns
    cols = ['date', 'state', 'district', 'total_enrolments', 'risk_score', 
            'risk_level', 'anomaly_score', 'risk_explanation']
    cols = [c for c in cols if c in anomalies.columns]
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        anomalies[cols].to_excel(writer, sheet_name='Anomalies', index=False)
        
        # Summary
        summary = pd.DataFrame({
            'Metric': ['Total Anomalies', 'Anomaly Rate', 'Avg Anomaly Score', 'States Affected'],
            'Value': [
                len(df[df['is_anomaly']]),
                f"{df['is_anomaly'].mean() * 100:.2f}%",
                f"{df[df['is_anomaly']]['anomaly_score'].mean():.3f}" if 'anomaly_score' in df.columns else 'N/A',
                df[df['is_anomaly']]['state'].nunique() if 'state' in df.columns else 'N/A'
            ]
        })
        summary.to_excel(writer, sheet_name='Summary', index=False)
    
    return output_path


def generate_text_report(
    df: pd.DataFrame,
    title: str = "UIDAI Aadhaar Data Vault - Analysis Report"
) -> str:
    """
    Generate a text-based summary report.
    
    Args:
        df: DataFrame with analysis results
        title: Report title
        
    Returns:
        Report as string
    """
    report_lines = [
        "=" * 60,
        title,
        "=" * 60,
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n" + "-" * 40,
        "DATA SUMMARY",
        "-" * 40,
    ]
    
    # Basic stats
    report_lines.append(f"Total Records: {len(df):,}")
    
    if 'date' in df.columns:
        report_lines.append(f"Date Range: {df['date'].min()} to {df['date'].max()}")
    
    if 'state' in df.columns:
        report_lines.append(f"States: {df['state'].nunique()}")
    
    if 'district' in df.columns:
        report_lines.append(f"Districts: {df['district'].nunique()}")
    
    if 'total_enrolments' in df.columns:
        report_lines.append(f"Total Enrolments: {df['total_enrolments'].sum():,.0f}")
    
    # Risk analysis
    if 'risk_level' in df.columns:
        report_lines.extend([
            "\n" + "-" * 40,
            "RISK ANALYSIS",
            "-" * 40,
        ])
        risk_dist = df['risk_level'].value_counts()
        for level, count in risk_dist.items():
            pct = count / len(df) * 100
            report_lines.append(f"  {level}: {count} ({pct:.1f}%)")
    
    # Anomaly detection
    if 'is_anomaly' in df.columns:
        report_lines.extend([
            "\n" + "-" * 40,
            "ANOMALY DETECTION",
            "-" * 40,
        ])
        anomaly_count = df['is_anomaly'].sum()
        report_lines.append(f"Anomalies Detected: {anomaly_count}")
        report_lines.append(f"Anomaly Rate: {anomaly_count / len(df) * 100:.2f}%")
        
        if anomaly_count > 0 and 'state' in df.columns:
            top_states = df[df['is_anomaly']]['state'].value_counts().head(5)
            report_lines.append("\nTop Affected States:")
            for state, count in top_states.items():
                report_lines.append(f"  - {state}: {count} anomalies")
    
    # High risk regions
    if 'risk_score' in df.columns and 'state' in df.columns:
        report_lines.extend([
            "\n" + "-" * 40,
            "HIGH RISK REGIONS",
            "-" * 40,
        ])
        high_risk = df.groupby('state')['risk_score'].mean().nlargest(10)
        for state, score in high_risk.items():
            report_lines.append(f"  {state}: {score:.3f}")
    
    report_lines.extend([
        "\n" + "=" * 60,
        "END OF REPORT",
        "=" * 60,
    ])
    
    return "\n".join(report_lines)


def save_report(
    content: str,
    output_path: str
) -> str:
    """
    Save text report to file.
    
    Args:
        content: Report content
        output_path: Path for output file
        
    Returns:
        Path to created file
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return output_path


def generate_html_report(
    df: pd.DataFrame,
    title: str = "UIDAI Aadhaar Data Vault - Analysis Report",
    include_charts: bool = False
) -> str:
    """
    Generate an HTML report with styling.
    
    Args:
        df: DataFrame with analysis results
        title: Report title
        include_charts: Whether to include embedded charts
        
    Returns:
        HTML content as string
    """
    # CSS styling
    css = """
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #1E3A5F; border-bottom: 3px solid #667eea; padding-bottom: 10px; }
        h2 { color: #333; margin-top: 30px; }
        .metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }
        .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; }
        .metric-value { font-size: 2em; font-weight: bold; }
        .metric-label { font-size: 0.9em; opacity: 0.9; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #667eea; color: white; }
        tr:hover { background: #f5f5f5; }
        .risk-high { background: #ffcccc; }
        .risk-medium { background: #fff3cd; }
        .risk-low { background: #d4edda; }
        .footer { text-align: center; margin-top: 30px; color: #888; font-size: 0.9em; }
    </style>
    """
    
    # Build HTML
    html_parts = [
        "<!DOCTYPE html>",
        "<html><head>",
        f"<title>{title}</title>",
        css,
        "</head><body>",
        '<div class="container">',
        f"<h1>🏛️ {title}</h1>",
        f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
    ]
    
    # Metrics grid
    html_parts.append('<div class="metric-grid">')
    
    metrics = [
        ('Total Records', f"{len(df):,}"),
        ('States', str(df['state'].nunique()) if 'state' in df.columns else 'N/A'),
        ('Districts', str(df['district'].nunique()) if 'district' in df.columns else 'N/A'),
        ('Total Enrolments', f"{df['total_enrolments'].sum():,.0f}" if 'total_enrolments' in df.columns else 'N/A'),
    ]
    
    for label, value in metrics:
        html_parts.append(f'''
            <div class="metric-card">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
        ''')
    
    html_parts.append('</div>')
    
    # Risk distribution
    if 'risk_level' in df.columns:
        html_parts.append("<h2>📊 Risk Distribution</h2>")
        html_parts.append("<table><tr><th>Risk Level</th><th>Count</th><th>Percentage</th></tr>")
        
        risk_dist = df['risk_level'].value_counts()
        for level, count in risk_dist.items():
            pct = count / len(df) * 100
            risk_class = f"risk-{level.lower()}" if level.lower() in ['high', 'medium', 'low'] else ''
            html_parts.append(f'<tr class="{risk_class}"><td>{level}</td><td>{count}</td><td>{pct:.1f}%</td></tr>')
        
        html_parts.append("</table>")
    
    # Top risk regions
    if 'risk_score' in df.columns and 'state' in df.columns:
        html_parts.append("<h2>⚠️ Top Risk Regions</h2>")
        html_parts.append("<table><tr><th>State</th><th>Avg Risk Score</th><th>Records</th></tr>")
        
        state_risk = df.groupby('state').agg({
            'risk_score': 'mean',
            'state': 'count'
        }).rename(columns={'state': 'count'}).nlargest(10, 'risk_score')
        
        for state, row in state_risk.iterrows():
            html_parts.append(f'<tr><td>{state}</td><td>{row["risk_score"]:.3f}</td><td>{row["count"]}</td></tr>')
        
        html_parts.append("</table>")
    
    # Footer
    html_parts.extend([
        '<div class="footer">',
        'UIDAI Data Hackathon 2026 | Mini Aadhaar Data Vault',
        '</div>',
        '</div></body></html>'
    ])
    
    return "\n".join(html_parts)


if __name__ == "__main__":
    # Test export functions
    print("Testing export module...")
    
    sample_data = pd.DataFrame({
        'date': pd.date_range('2025-01-01', periods=10),
        'state': ['Maharashtra'] * 5 + ['Karnataka'] * 5,
        'district': ['Mumbai'] * 5 + ['Bangalore'] * 5,
        'total_enrolments': [1000, 1200, 900, 1100, 1150, 800, 850, 900, 950, 1000],
        'risk_score': [0.3, 0.4, 0.8, 0.5, 0.6, 0.2, 0.3, 0.7, 0.4, 0.5],
        'risk_level': ['Low', 'Medium', 'High', 'Medium', 'Medium', 'Low', 'Low', 'High', 'Medium', 'Medium'],
        'is_anomaly': [False, False, True, False, False, False, False, True, False, False]
    })
    
    text_report = generate_text_report(sample_data)
    print(text_report)
