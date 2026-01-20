"""
Geographic Visualization Module
Creates India choropleth maps with state-wise risk and enrolment visualization.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import requests
from typing import Optional, Dict

# India GeoJSON URL (state boundaries)
INDIA_GEOJSON_URL = "https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson"

# State name mapping (to handle naming differences between data and GeoJSON)
STATE_NAME_MAPPING = {
    'Andaman And Nicobar Islands': 'Andaman & Nicobar Island',
    'Andaman & Nicobar': 'Andaman & Nicobar Island',
    'Dadra And Nagar Haveli': 'Dadara & Nagar Havelli',
    'Dadra & Nagar Haveli': 'Dadara & Nagar Havelli',
    'Daman And Diu': 'Daman & Diu',
    'Jammu And Kashmir': 'Jammu & Kashmir',
    'NCT Of Delhi': 'NCT of Delhi',
    'Delhi': 'NCT of Delhi',
    'Telengana': 'Telangana',
    'Chattisgarh': 'Chhattisgarh',
    'Pondicherry': 'Puducherry',
    'Orissa': 'Odisha'
}


def load_india_geojson() -> dict:
    """
    Load India GeoJSON from URL.
    
    Returns:
        GeoJSON dictionary with state boundaries
    """
    try:
        response = requests.get(INDIA_GEOJSON_URL, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error loading GeoJSON: {e}")
        return None


def normalize_state_name(state: str) -> str:
    """
    Normalize state names to match GeoJSON.
    
    Args:
        state: State name from data
        
    Returns:
        Normalized state name
    """
    state = state.strip().title()
    return STATE_NAME_MAPPING.get(state, state)


def create_risk_heatmap(df: pd.DataFrame,
                        state_col: str = 'state',
                        value_col: str = 'risk_score',
                        title: str = 'India Risk Heatmap') -> go.Figure:
    """
    Create an India choropleth map showing risk scores by state.
    
    Args:
        df: DataFrame with state-level data
        state_col: Column name for state
        value_col: Column name for value to visualize
        title: Map title
        
    Returns:
        Plotly Figure object
    """
    # Load GeoJSON
    geojson = load_india_geojson()
    if geojson is None:
        # Return empty figure with error message
        fig = go.Figure()
        fig.add_annotation(text="Unable to load India map data", 
                          xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    # Aggregate data by state
    state_data = df.groupby(state_col).agg({
        value_col: 'mean',
        'total_enrolments': 'sum' if 'total_enrolments' in df.columns else 'count'
    }).reset_index()
    
    # Normalize state names
    state_data['state_normalized'] = state_data[state_col].apply(normalize_state_name)
    
    # Create choropleth map
    fig = px.choropleth(
        state_data,
        geojson=geojson,
        locations='state_normalized',
        featureidkey='properties.ST_NM',
        color=value_col,
        color_continuous_scale='RdYlGn_r',  # Red = high risk, Green = low risk
        range_color=[0, 1],
        hover_name=state_col,
        hover_data={
            value_col: ':.3f',
            'total_enrolments': ':,.0f',
            'state_normalized': False
        },
        title=title
    )
    
    # Focus on India
    fig.update_geos(
        visible=False,
        fitbounds='locations',
        bgcolor='rgba(0,0,0,0)'
    )
    
    # Update layout
    fig.update_layout(
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        coloraxis_colorbar={
            'title': 'Risk Score',
            'tickvals': [0, 0.25, 0.5, 0.75, 1],
            'ticktext': ['Low', '', 'Medium', '', 'High']
        },
        paper_bgcolor='rgba(0,0,0,0)',
        geo=dict(bgcolor='rgba(0,0,0,0)')
    )
    
    return fig


def create_enrolment_heatmap(df: pd.DataFrame,
                            state_col: str = 'state',
                            title: str = 'Enrolment Distribution by State') -> go.Figure:
    """
    Create an India choropleth map showing enrolment distribution.
    
    Args:
        df: DataFrame with state-level data
        state_col: Column name for state
        title: Map title
        
    Returns:
        Plotly Figure object
    """
    geojson = load_india_geojson()
    if geojson is None:
        fig = go.Figure()
        fig.add_annotation(text="Unable to load India map data", 
                          xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    # Aggregate total enrolments by state
    if 'total_enrolments' not in df.columns:
        df = df.copy()
        df['total_enrolments'] = df['age_0_5'] + df['age_5_17'] + df['age_18_greater']
    
    state_data = df.groupby(state_col).agg({
        'total_enrolments': 'sum'
    }).reset_index()
    
    # Normalize state names
    state_data['state_normalized'] = state_data[state_col].apply(normalize_state_name)
    
    # Log scale for better visualization (enrolments vary widely)
    state_data['log_enrolments'] = np.log10(state_data['total_enrolments'] + 1)
    
    fig = px.choropleth(
        state_data,
        geojson=geojson,
        locations='state_normalized',
        featureidkey='properties.ST_NM',
        color='log_enrolments',
        color_continuous_scale='Blues',
        hover_name=state_col,
        hover_data={
            'total_enrolments': ':,.0f',
            'log_enrolments': False,
            'state_normalized': False
        },
        title=title
    )
    
    fig.update_geos(
        visible=False,
        fitbounds='locations',
        bgcolor='rgba(0,0,0,0)'
    )
    
    fig.update_layout(
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        coloraxis_colorbar={
            'title': 'Enrolments (log scale)'
        },
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_anomaly_heatmap(df: pd.DataFrame,
                          state_col: str = 'state',
                          title: str = 'Anomaly Density by State') -> go.Figure:
    """
    Create an India choropleth map showing anomaly density.
    
    Args:
        df: DataFrame with anomaly flags
        state_col: Column name for state
        title: Map title
        
    Returns:
        Plotly Figure object
    """
    geojson = load_india_geojson()
    if geojson is None:
        fig = go.Figure()
        fig.add_annotation(text="Unable to load India map data", 
                          xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    # Calculate anomaly rate per state
    if 'is_anomaly' in df.columns:
        state_data = df.groupby(state_col).agg({
            'is_anomaly': ['sum', 'count']
        }).reset_index()
        state_data.columns = [state_col, 'anomaly_count', 'total_records']
        state_data['anomaly_rate'] = state_data['anomaly_count'] / state_data['total_records']
    else:
        # No anomaly data, return empty map
        fig = go.Figure()
        fig.add_annotation(text="No anomaly data available", 
                          xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    state_data['state_normalized'] = state_data[state_col].apply(normalize_state_name)
    
    fig = px.choropleth(
        state_data,
        geojson=geojson,
        locations='state_normalized',
        featureidkey='properties.ST_NM',
        color='anomaly_rate',
        color_continuous_scale='Reds',
        range_color=[0, state_data['anomaly_rate'].quantile(0.95)],
        hover_name=state_col,
        hover_data={
            'anomaly_count': ':,.0f',
            'anomaly_rate': ':.1%',
            'state_normalized': False
        },
        title=title
    )
    
    fig.update_geos(
        visible=False,
        fitbounds='locations',
        bgcolor='rgba(0,0,0,0)'
    )
    
    fig.update_layout(
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        coloraxis_colorbar={
            'title': 'Anomaly Rate',
            'tickformat': '.0%'
        },
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_demographic_heatmap(df: pd.DataFrame,
                              state_col: str = 'state',
                              age_group: str = 'child_ratio',
                              title: str = 'Child Enrolment Ratio by State') -> go.Figure:
    """
    Create an India choropleth map showing demographic distribution.
    
    Args:
        df: DataFrame with demographic data
        state_col: Column name for state
        age_group: Which ratio to show ('child_ratio' or 'adult_ratio')
        title: Map title
        
    Returns:
        Plotly Figure object
    """
    geojson = load_india_geojson()
    if geojson is None:
        fig = go.Figure()
        fig.add_annotation(text="Unable to load India map data", 
                          xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    df = df.copy()
    
    # Calculate ratios if not present
    if 'total_enrolments' not in df.columns:
        df['total_enrolments'] = df['age_0_5'] + df['age_5_17'] + df['age_18_greater']
    
    if 'child_ratio' not in df.columns:
        df['child_ratio'] = (df['age_0_5'] + df['age_5_17']) / df['total_enrolments'].replace(0, np.nan)
    
    state_data = df.groupby(state_col).agg({
        'age_0_5': 'sum',
        'age_5_17': 'sum',
        'age_18_greater': 'sum',
        'total_enrolments': 'sum'
    }).reset_index()
    
    state_data['child_ratio'] = (state_data['age_0_5'] + state_data['age_5_17']) / state_data['total_enrolments']
    state_data['state_normalized'] = state_data[state_col].apply(normalize_state_name)
    
    fig = px.choropleth(
        state_data,
        geojson=geojson,
        locations='state_normalized',
        featureidkey='properties.ST_NM',
        color='child_ratio',
        color_continuous_scale='Viridis',
        range_color=[0, 1],
        hover_name=state_col,
        hover_data={
            'child_ratio': ':.1%',
            'total_enrolments': ':,.0f',
            'state_normalized': False
        },
        title=title
    )
    
    fig.update_geos(
        visible=False,
        fitbounds='locations',
        bgcolor='rgba(0,0,0,0)'
    )
    
    fig.update_layout(
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        coloraxis_colorbar={
            'title': 'Child Ratio',
            'tickformat': '.0%'
        },
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


if __name__ == "__main__":
    # Test the geographic visualization
    print("Testing geographic visualization module...")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'state': ['Maharashtra', 'Karnataka', 'Tamil Nadu', 'Uttar Pradesh', 'Gujarat'],
        'total_enrolments': [1000000, 800000, 750000, 1200000, 600000],
        'risk_score': [0.3, 0.7, 0.4, 0.8, 0.2],
        'is_anomaly': [False, True, False, True, False]
    })
    
    print("\nLoading India GeoJSON...")
    geojson = load_india_geojson()
    if geojson:
        print(f"  Loaded {len(geojson['features'])} state boundaries")
    
    print("\nCreating risk heatmap...")
    fig = create_risk_heatmap(sample_data)
    print("  Risk heatmap created successfully")
    
    # Save to HTML for testing
    fig.write_html("test_heatmap.html")
    print("  Saved to test_heatmap.html")
