"""
UIDAI Aadhaar Data Vault - Streamlit Dashboard
Interactive visualization and analysis platform for Aadhaar enrolment data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import load_all_data, get_data_summary
from src.data.preprocessor import clean_data, aggregate_by_state, aggregate_by_district, get_daily_totals
from src.features.temporal import calculate_growth_rate, calculate_volatility, detect_trends, calculate_seasonal_indicator
from src.features.demographic import calculate_age_imbalance, calculate_adult_child_ratio, get_demographic_summary
from src.features.risk_scoring import calculate_risk_score, classify_risk_level, get_high_risk_regions
from src.models.anomaly_detector import detect_anomalies, get_anomaly_summary
from src.models.statistical import detect_statistical_outliers, get_distribution_stats
from src.visualization.geographic import (
    create_risk_heatmap, 
    create_enrolment_heatmap, 
    create_anomaly_heatmap, 
    create_demographic_heatmap
)
from src.models.forecasting import forecast_total_enrolments
from src.utils.alerts import AlertEngine
from src.utils.export import export_to_excel, generate_html_report

# Page configuration
st.set_page_config(
    page_title="UIDAI Aadhaar Data Vault",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #1E3A5F 0%, #2E5A8F 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 1rem;
    }
    .stMetric > div {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    .alert-card {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        border-left: 5px solid;
    }
    .alert-critical { background-color: #ffebee; border-left-color: #f44336; }
    .alert-warning { background-color: #fffde7; border-left-color: #fbc02d; }
    .alert-info { background-color: #e3f2fd; border-left-color: #2196f3; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_and_process_data():
    """Load and preprocess all data."""
    with st.spinner("Loading data..."):
        df = load_all_data()
        df = clean_data(df)
        return df


@st.cache_data(ttl=3600)
def prepare_analysis_data(_df):
    """Prepare data with all features for analysis."""
    # Aggregate by state
    state_df = aggregate_by_state(_df)
    
    # Calculate features
    state_df = calculate_growth_rate(state_df, group_cols=['state'])
    state_df = calculate_volatility(state_df, group_cols=['state'])
    state_df = calculate_age_imbalance(state_df)
    state_df = calculate_adult_child_ratio(state_df)
    state_df = detect_trends(state_df, group_cols=['state'])
    
    # Calculate risk scores
    state_df = calculate_risk_score(state_df)
    state_df = classify_risk_level(state_df)
    
    return state_df


def render_header():
    """Render the dashboard header."""
    st.markdown('<h1 class="main-header">🏛️ Mini Aadhaar Data Vault</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">ML-Driven Anomaly Detection & Risk Intelligence for UIDAI</p>', unsafe_allow_html=True)
    st.divider()


def render_kpis(df, state_df):
    """Render key performance indicators."""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_enrolments = (df['age_0_5'] + df['age_5_17'] + df['age_18_greater']).sum()
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Total Enrolments", f"{total_enrolments:,.0f}")
    with col3:
        st.metric("States", df['state'].nunique())
    with col4:
        st.metric("Districts", df['district'].nunique())
    with col5:
        anomaly_rate = state_df['is_anomaly'].mean() * 100 if 'is_anomaly' in state_df.columns else 0
        st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")


def render_geographic_maps(state_df, df):
    """Render geographic heatmaps of India."""
    st.subheader("🗺️ Geographic Analysis")
    
    map_type = st.radio(
        "Select Map View",
        ["Risk Score", "Enrolments", "Anomaly Density", "Child Ratio"],
        key="map_view_radio",
        horizontal=True
    )
    
    with st.spinner("Loading India map..."):
        if map_type == "Risk Score":
            fig = create_risk_heatmap(state_df, title="India State-wise Risk Heatmap")
        elif map_type == "Enrolments":
            fig = create_enrolment_heatmap(state_df, title="Enrolment Distribution Across India")
        elif map_type == "Anomaly Density":
            fig = create_anomaly_heatmap(state_df, title="Anomaly Density Map")
        else:  # Child Ratio
            fig = create_demographic_heatmap(state_df, title="Child Enrolment Ratio Map")
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)


def render_trend_analysis(state_df):
    """Render trend analysis charts."""
    st.subheader("📈 Enrolment Trends Over Time")
    
    daily_df = state_df.groupby('date').agg({
        'total_enrolments': 'sum',
        'age_0_5': 'sum',
        'age_5_17': 'sum',
        'age_18_greater': 'sum'
    }).reset_index()
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(daily_df, x='date', y='total_enrolments', title='Daily Total Enrolments')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.area(daily_df, x='date', y=['age_0_5', 'age_5_17', 'age_18_greater'], title='Enrolments by Age Group')
        st.plotly_chart(fig, use_container_width=True)


def render_forecast_tab(df):
    """Render forecasting dashboard."""
    st.subheader("🔮 Predictive Forecasting")
    
    with st.spinner("Generating forecast..."):
        historical, forecast = forecast_total_enrolments(df, periods=30)
        
        fig = go.Figure()
        
        # Historical
        fig.add_trace(go.Scatter(
            x=historical['date'], y=historical['total_enrolments'],
            mode='lines', name='Historical', line=dict(color='blue')
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast['date'], y=forecast['predicted'],
            mode='lines', name='Forecast', line=dict(color='red', dash='dash')
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast['date'], forecast['date'][::-1]]),
            y=pd.concat([forecast['upper_bound'], forecast['lower_bound'][::-1]]),
            fill='toself', fillcolor='rgba(255,0,0,0.1)', line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence Interval'
        ))
        
        fig.update_layout(
            title='Total Enrolments Forecast (Next 30 Days)',
            xaxis_title='Date', yaxis_title='Enrolments',
            hovermode='x unified', height=500
        )
        st.plotly_chart(fig, use_container_width=True)


def render_alerts_tab(state_df):
    """Render alert system dashboard."""
    st.subheader("🔔 Real-time Alerts")
    
    engine = AlertEngine()
    alerts = engine.process(state_df)
    
    if not alerts:
        st.success("✅ No active alerts detected.")
        return
    
    summary = engine.get_summary()
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Critical Alerts", summary['critical'], delta_color="inverse")
    with col2: st.metric("Warnings", summary['warnings'])
    with col3: st.metric("Info", summary['info'])
    
    for alert in alerts[:20]:
        severity_class = f"alert-{alert.severity.value}"
        st.markdown(f"""
        <div class="alert-card {severity_class}">
            <strong>{alert.region}</strong> | {alert.timestamp.strftime('%y-%m-%d') if hasattr(alert.timestamp, 'strftime') else str(alert.timestamp)} <br>
            {alert.message}
        </div>
        """, unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar controls."""
    st.sidebar.title("🎛️ Dashboard Controls")
    
    st.sidebar.subheader("ML Models")
    run_anomaly = st.sidebar.checkbox("Isolation Forest Detection", value=True)
    contamination = st.sidebar.slider("Anomaly Threshold", 0.01, 0.20, 0.05, 0.01)
    
    st.sidebar.subheader("Downloads")
    if st.sidebar.button("📊 Export Excel Report"):
        st.sidebar.success("Export started...")
        
    st.sidebar.divider()
    st.sidebar.caption("UIDAI Data Hackathon 2026")
    
    return {'run_anomaly': run_anomaly, 'contamination': contamination}


def main():
    """Main dash entry."""
    render_header()
    options = render_sidebar()
    
    try:
        df = load_and_process_data()
        state_df = prepare_analysis_data(df)
        
        if options['run_anomaly']:
            feature_cols = ['total_enrolments', 'growth_rate', 'volatility_cv', 'demographic_imbalance']
            feature_cols = [c for c in feature_cols if c in state_df.columns]
            state_df = detect_anomalies(state_df, feature_cols, contamination=options['contamination'])
        
        render_kpis(df, state_df)
        
        tabs = st.tabs(["🗺️ India Map", "📈 Trends", "🚨 Anomalies", "⚠️ Risk", "🔮 Forecast", "🔔 Alerts", "👥 Demographics"])
        
        with tabs[0]: render_geographic_maps(state_df, df)
        with tabs[1]: render_trend_analysis(state_df)
        with tabs[2]: render_anomaly_detection(state_df)
        with tabs[3]: render_risk_dashboard(state_df)
        with tabs[4]: render_forecast_tab(state_df)
        with tabs[5]: render_alerts_tab(state_df)
        with tabs[6]: render_demographic_analysis(df)
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.exception(e)


def render_anomaly_detection(state_df):
    """Anomalies view."""
    st.subheader("🚨 Anomaly Detection Details")
    if 'is_anomaly' not in state_df.columns:
        st.info("Anomaly detection is disabled.")
        return
    anomalies = state_df[state_df['is_anomaly']]
    fig = px.scatter(state_df, x='total_enrolments', y='anomaly_score', color='is_anomaly', 
                     hover_data=['state', 'date'], title="Anomaly Score Distribution")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(anomalies.sort_values('anomaly_score', ascending=False).head(20))


def render_risk_dashboard(state_df):
    """Risk view."""
    st.subheader("⚠️ Risk Intelligence")
    risk_counts = state_df['risk_level'].value_counts()
    fig = px.pie(values=risk_counts.values, names=risk_counts.index, title='Risk distribution')
    st.plotly_chart(fig, use_container_width=True)


def render_demographic_analysis(df):
    """Demographics view."""
    st.subheader("👥 Demographic Skew")
    totals = [df['age_0_5'].sum(), df['age_5_17'].sum(), df['age_18_greater'].sum()]
    fig = px.pie(values=totals, names=['0-5', '5-17', '18+'], title='Age distribution')
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
