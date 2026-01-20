"""
Geographic Mapping Module
India state-level choropleth heatmaps for risk visualization.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Dict
import json


# India state name mapping to GeoJSON feature names
STATE_NAME_MAPPING = {
    'Andaman and Nicobar Islands': 'Andaman & Nicobar Island',
    'Andhra Pradesh': 'Andhra Pradesh',
    'Arunachal Pradesh': 'Arunachal Pradesh',
    'Assam': 'Assam',
    'Bihar': 'Bihar',
    'Chandigarh': 'Chandigarh',
    'Chhattisgarh': 'Chhattisgarh',
    'Dadra and Nagar Haveli': 'Dadara & Nagar Haveli',
    'Daman and Diu': 'Daman & Diu',
    'Delhi': 'NCT of Delhi',
    'Goa': 'Goa',
    'Gujarat': 'Gujarat',
    'Haryana': 'Haryana',
    'Himachal Pradesh': 'Himachal Pradesh',
    'Jammu and Kashmir': 'Jammu & Kashmir',
    'Jharkhand': 'Jharkhand',
    'Karnataka': 'Karnataka',
    'Kerala': 'Kerala',
    'Ladakh': 'Ladakh',
    'Lakshadweep': 'Lakshadweep',
    'Madhya Pradesh': 'Madhya Pradesh',
    'Maharashtra': 'Maharashtra',
    'Manipur': 'Manipur',
    'Meghalaya': 'Meghalaya',
    'Mizoram': 'Mizoram',
    'Nagaland': 'Nagaland',
    'Odisha': 'Odisha',
    'Puducherry': 'Puducherry',
    'Punjab': 'Punjab',
    'Rajasthan': 'Rajasthan',
    'Sikkim': 'Sikkim',
    'Tamil Nadu': 'Tamil Nadu',
    'Telangana': 'Telangana',
    'Tripura': 'Tripura',
    'Uttar Pradesh': 'Uttar Pradesh',
    'Uttarakhand': 'Uttarakhand',
    'West Bengal': 'West Bengal',
}

# India GeoJSON URL (public resource)
INDIA_GEOJSON_URL = "https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson"


def load_india_geojson() -> dict:
    """
    Load India states GeoJSON from public URL.
    
    Returns:
        GeoJSON dictionary
    """
    import urllib.request
    
    try:
        with urllib.request.urlopen(INDIA_GEOJSON_URL) as response:
            geojson = json.loads(response.read().decode())
        return geojson
    except Exception as e:
        print(f"Warning: Could not load GeoJSON: {e}")
        return None


def create_state_choropleth(
    df: pd.DataFrame,
    value_col: str = 'risk_score',
    state_col: str = 'state',
    title: str = 'State-wise Risk Heatmap',
    color_scale: str = 'RdYlGn_r',
    hover_data: Optional[list] = None
) -> go.Figure:
    """
    Create a choropleth map of India states.
    
    Args:
        df: DataFrame with state-level data
        value_col: Column to visualize
        state_col: Column with state names
        title: Chart title
        color_scale: Plotly color scale
        hover_data: Additional columns for hover info
        
    Returns:
        Plotly Figure object
    """
    geojson = load_india_geojson()
    
    if geojson is None:
        # Fallback to simple bar chart if GeoJSON fails
        fig = px.bar(
            df.nlargest(15, value_col),
            x=state_col, y=value_col,
            title=f"{title} (Map unavailable - showing bar chart)",
            color=value_col,
            color_continuous_scale=color_scale
        )
        return fig
    
    # Prepare data - aggregate by state if needed
    if 'date' in df.columns:
        state_data = df.groupby(state_col).agg({
            value_col: 'mean',
            'total_enrolments': 'sum' if 'total_enrolments' in df.columns else 'first'
        }).reset_index()
    else:
        state_data = df.copy()
    
    # Apply state name mapping for GeoJSON compatibility
    from src.utils.geo_mapping import STATE_NAME_MAPPING
    state_data[state_col] = state_data[state_col].map(lambda x: STATE_NAME_MAPPING.get(x, x))
    
    # Create the choropleth
    fig = px.choropleth(
        state_data,
        geojson=geojson,
        locations=state_col,
        featureidkey="properties.ST_NM",
        color=value_col,
        color_continuous_scale=color_scale,
        title=title,
        hover_name=state_col,
        hover_data=hover_data if hover_data else [value_col]
    )
    
    fig.update_geos(
        visible=False,
        fitbounds="locations",
        bgcolor='rgba(0,0,0,0)'
    )
    
    fig.update_layout(
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        geo=dict(
            showframe=False,
            showcoastlines=False,
        ),
        height=500
    )
    
    return fig


def create_risk_heatmap_grid(
    df: pd.DataFrame,
    x_col: str = 'state',
    y_col: str = 'date',
    value_col: str = 'risk_score',
    title: str = 'Risk Heatmap Over Time'
) -> go.Figure:
    """
    Create a grid heatmap showing risk over states and time.
    
    Args:
        df: DataFrame with time-series state data
        x_col: Column for x-axis
        y_col: Column for y-axis
        value_col: Column for heatmap values
        title: Chart title
        
    Returns:
        Plotly Figure object
    """
    # Pivot data for heatmap
    pivot_df = df.pivot_table(
        values=value_col,
        index=y_col,
        columns=x_col,
        aggfunc='mean'
    ).fillna(0)
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns,
        y=pivot_df.index,
        colorscale='RdYlGn_r',
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='State',
        yaxis_title='Date',
        height=600,
        xaxis={'tickangle': -45}
    )
    
    return fig


def create_district_bubble_map(
    df: pd.DataFrame,
    size_col: str = 'total_enrolments',
    color_col: str = 'risk_score',
    title: str = 'District Risk Bubble Chart'
) -> go.Figure:
    """
    Create a bubble chart showing district-level data.
    
    Args:
        df: DataFrame with district-level data
        size_col: Column for bubble size
        color_col: Column for bubble color
        title: Chart title
        
    Returns:
        Plotly Figure object
    """
    # Aggregate by district
    district_data = df.groupby(['state', 'district']).agg({
        size_col: 'sum' if size_col == 'total_enrolments' else 'mean',
        color_col: 'mean'
    }).reset_index()
    
    # Sort and take top 50 for readability
    district_data = district_data.nlargest(50, size_col)
    
    fig = px.scatter(
        district_data,
        x='state',
        y=color_col,
        size=size_col,
        color=color_col,
        hover_name='district',
        color_continuous_scale='RdYlGn_r',
        title=title,
        size_max=40
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=500
    )
    
    return fig


if __name__ == "__main__":
    # Test the geographic mapping
    print("Testing geographic mapping module...")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'state': ['Maharashtra', 'Karnataka', 'Tamil Nadu', 'Uttar Pradesh', 'Gujarat'],
        'risk_score': [0.75, 0.45, 0.30, 0.85, 0.55],
        'total_enrolments': [50000, 40000, 35000, 60000, 30000]
    })
    
    fig = create_state_choropleth(sample_data)
    print("Choropleth created successfully!")
    fig.show()
