"""
Chart builder components using Plotly
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import List, Optional, Dict, Any


def create_line_chart(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    color: Optional[str] = None,
    height: int = 400
) -> go.Figure:
    """
    Create a line chart
    
    Args:
        data: DataFrame with data
        x: X-axis column
        y: Y-axis column
        title: Chart title
        color: Optional column for color grouping
        height: Chart height
        
    Returns:
        Plotly figure
    """
    if color:
        fig = px.line(data, x=x, y=y, color=color, title=title)
    else:
        fig = px.line(data, x=x, y=y, title=title)
    
    fig.update_layout(
        height=height,
        hovermode='x unified',
        showlegend=True
    )
    
    return fig


def create_bar_chart(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    color: Optional[str] = None,
    orientation: str = 'v',
    height: int = 400
) -> go.Figure:
    """
    Create a bar chart
    
    Args:
        data: DataFrame with data
        x: X-axis column
        y: Y-axis column
        title: Chart title
        color: Optional column for color grouping
        orientation: 'v' for vertical, 'h' for horizontal
        height: Chart height
        
    Returns:
        Plotly figure
    """
    if color:
        fig = px.bar(data, x=x, y=y, color=color, title=title, orientation=orientation)
    else:
        fig = px.bar(data, x=x, y=y, title=title, orientation=orientation)
    
    fig.update_layout(height=height)
    
    return fig


def create_scatter_chart(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    size: Optional[str] = None,
    color: Optional[str] = None,
    hover_name: Optional[str] = None,
    height: int = 400
) -> go.Figure:
    """
    Create a scatter plot
    
    Args:
        data: DataFrame with data
        x: X-axis column
        y: Y-axis column
        title: Chart title
        size: Optional column for bubble size
        color: Optional column for color
        hover_name: Optional column for hover labels
        height: Chart height
        
    Returns:
        Plotly figure
    """
    fig = px.scatter(
        data,
        x=x,
        y=y,
        size=size,
        color=color,
        hover_name=hover_name,
        title=title
    )
    
    fig.update_layout(height=height)
    
    return fig


def create_pie_chart(
    data: pd.DataFrame,
    values: str,
    names: str,
    title: str,
    height: int = 400
) -> go.Figure:
    """
    Create a pie chart
    
    Args:
        data: DataFrame with data
        values: Column with values
        names: Column with labels
        title: Chart title
        height: Chart height
        
    Returns:
        Plotly figure
    """
    fig = px.pie(data, values=values, names=names, title=title)
    
    fig.update_layout(height=height)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    return fig


def create_time_series_chart(
    data: pd.DataFrame,
    date_column: str,
    value_columns: List[str],
    title: str,
    height: int = 400
) -> go.Figure:
    """
    Create a multi-line time series chart
    
    Args:
        data: DataFrame with data
        date_column: Column with dates
        value_columns: List of columns to plot
        title: Chart title
        height: Chart height
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    for col in value_columns:
        if col in data.columns:
            fig.add_trace(go.Scatter(
                x=data[date_column],
                y=data[col],
                mode='lines+markers',
                name=col
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title=date_column,
        yaxis_title="Value",
        height=height,
        hovermode='x unified'
    )
    
    return fig


def create_grouped_bar_chart(
    data: pd.DataFrame,
    categories: str,
    values: List[str],
    title: str,
    height: int = 400
) -> go.Figure:
    """
    Create a grouped bar chart
    
    Args:
        data: DataFrame with data
        categories: Column for categories (x-axis)
        values: List of columns to plot as bars
        title: Chart title
        height: Chart height
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    for val in values:
        if val in data.columns:
            fig.add_trace(go.Bar(
                name=val,
                x=data[categories],
                y=data[val]
            ))
    
    fig.update_layout(
        title=title,
        barmode='group',
        height=height
    )
    
    return fig


def create_funnel_chart(
    data: pd.DataFrame,
    stages: str,
    values: str,
    title: str,
    height: int = 400
) -> go.Figure:
    """
    Create a funnel chart (useful for conversion tracking)
    
    Args:
        data: DataFrame with data
        stages: Column with stage names
        values: Column with values
        title: Chart title
        height: Chart height
        
    Returns:
        Plotly figure
    """
    fig = go.Figure(go.Funnel(
        y=data[stages],
        x=data[values],
        textposition="inside",
        textinfo="value+percent initial"
    ))
    
    fig.update_layout(
        title=title,
        height=height
    )
    
    return fig


def create_gauge_chart(
    value: float,
    max_value: float,
    title: str,
    thresholds: Optional[Dict[str, float]] = None
) -> go.Figure:
    """
    Create a gauge chart (useful for metrics)
    
    Args:
        value: Current value
        max_value: Maximum value
        title: Chart title
        thresholds: Optional dict of threshold levels
        
    Returns:
        Plotly figure
    """
    if thresholds is None:
        thresholds = {
            'low': 0.3 * max_value,
            'medium': 0.7 * max_value,
            'high': max_value
        }
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [None, max_value]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, thresholds['low']], 'color': "#ef4444"},
                {'range': [thresholds['low'], thresholds['medium']], 'color': "#f59e0b"},
                {'range': [thresholds['medium'], max_value], 'color': "#10b981"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(height=300)
    
    return fig
