# Reusable UI and plotting helpers
"""
Utility functions for Streamlit app
"""
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List

API_URL = "http://127.0.0.1:8000"


def call_api(endpoint: str, method: str = "GET", data: Dict = None, files: Dict = None) -> tuple:
    """
    Call FastAPI backend
    
    Args:
        endpoint: API endpoint path
        method: HTTP method (GET or POST)
        data: JSON data for POST requests
        files: Files for upload
        
    Returns:
        Tuple of (success, response_data)
    """
    try:
        url = f"{API_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, timeout=30)
        elif method == "POST":
            if files:
                response = requests.post(url, files=files, data=data, timeout=30)
            else:
                response = requests.post(url, json=data, timeout=30)
        else:
            return False, {"error": "Unsupported HTTP method"}
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {"error": response.text, "status_code": response.status_code}
            
    except requests.exceptions.Timeout:
        return False, {"error": "Request timeout. The operation took too long."}
    except requests.exceptions.ConnectionError:
        return False, {"error": "Cannot connect to API. Make sure the backend is running."}
    except Exception as e:
        return False, {"error": str(e)}


def plot_shap_waterfall(shap_values: List[float], feature_names: List[str], base_value: float):
    """Create SHAP waterfall plot using Plotly"""
    
    # Sort by absolute value
    sorted_indices = sorted(range(len(shap_values)), key=lambda i: abs(shap_values[i]), reverse=True)
    
    sorted_values = [shap_values[i] for i in sorted_indices]
    sorted_names = [feature_names[i] for i in sorted_indices]
    
    # Create waterfall data
    cumsum = base_value
    y_values = [base_value]
    
    for val in sorted_values:
        cumsum += val
        y_values.append(cumsum)
    
    # Create figure
    fig = go.Figure()
    
    # Add bars
    colors = ['red' if v < 0 else 'green' for v in sorted_values]
    
    fig.add_trace(go.Waterfall(
        name="SHAP",
        orientation="v",
        measure=["relative"] * len(sorted_values) + ["total"],
        x=sorted_names + ["Prediction"],
        y=sorted_values + [cumsum - base_value],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "rgba(255, 50, 50, 0.7)"}},
        increasing={"marker": {"color": "rgba(50, 200, 50, 0.7)"}},
        totals={"marker": {"color": "rgba(50, 100, 200, 0.7)"}}
    ))
    
    fig.update_layout(
        title="SHAP Waterfall Plot - Feature Contributions",
        xaxis_title="Features",
        yaxis_title="SHAP Value",
        showlegend=False,
        height=500
    )
    
    return fig


def plot_shap_bar(feature_names: List[str], importance_scores: List[float]):
    """Create SHAP feature importance bar chart"""
    
    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_scores
    })
    
    fig = px.bar(
        df,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Global Feature Importance (SHAP)',
        labels={'Importance': 'Mean |SHAP Value|', 'Feature': ''},
        color='Importance',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(height=400, showlegend=False)
    
    return fig


def plot_bias_metrics(group_metrics: Dict[str, Dict]):
    """Create grouped bar chart for bias metrics"""
    
    groups = list(group_metrics.keys())
    
    metrics_to_plot = ['positive_rate', 'accuracy', 'true_positive_rate']
    metric_labels = ['Positive Rate', 'Accuracy', 'True Positive Rate']
    
    fig = go.Figure()
    
    for metric, label in zip(metrics_to_plot, metric_labels):
        values = [group_metrics[group].get(metric, 0) for group in groups]
        fig.add_trace(go.Bar(
            name=label,
            x=groups,
            y=values,
            text=[f'{v:.1%}' for v in values],
            textposition='auto'
        ))
    
    fig.update_layout(
        title='Performance Metrics by Group',
        xaxis_title='Group',
        yaxis_title='Score',
        barmode='group',
        height=500,
        yaxis=dict(tickformat='.0%')
    )
    
    return fig


def plot_fairness_gauge(disparate_impact: float, threshold: float = 0.8):
    """Create gauge chart for disparate impact"""
    
    color = "red" if disparate_impact < threshold else "green"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=disparate_impact,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Disparate Impact", 'font': {'size': 24}},
        delta={'reference': threshold, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, threshold], 'color': 'rgba(255, 0, 0, 0.3)'},
                {'range': [threshold, 1], 'color': 'rgba(0, 255, 0, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold
            }
        }
    ))
    
    fig.update_layout(height=300)
    
    return fig


def display_lime_table(explanations: List[Dict]):
    """Display LIME explanations as a table"""
    
    if not explanations:
        st.warning("No LIME explanations available")
        return
    
    explanation = explanations[0]
    feature_weights = explanation.get('feature_weights', [])
    
    if not feature_weights:
        st.warning("No feature weights in LIME explanation")
        return
    
    df = pd.DataFrame(feature_weights)
    df['weight'] = df['weight'].round(4)
    df['impact'] = df['weight'].apply(lambda x: '🔴 Negative' if x < 0 else '🟢 Positive')
    df = df.sort_values('weight', key=abs, ascending=False)
    
    st.dataframe(
        df,
        column_config={
            "feature": "Feature Condition",
            "weight": st.column_config.NumberColumn("Weight", format="%.4f"),
            "impact": "Impact Direction"
        },
        hide_index=True,
        use_container_width=True
    )


def create_summary_card(title: str, content: str, icon: str = "📝"):
    """Create a styled summary card"""
    
    st.markdown(f"""
    <div style="
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    ">
        <h3 style="margin-top: 0;">{icon} {title}</h3>
        <p style="white-space: pre-wrap; line-height: 1.6;">{content}</p>
    </div>
    """, unsafe_allow_html=True)


def parse_sample_input(text: str) -> Dict:
    """Parse comma-separated input into dictionary"""
    try:
        parts = [p.strip() for p in text.split(',')]
        
        # Expected format: age, income, loan_amount, credit_score, employment_years, debt_to_income
        if len(parts) != 6:
            return None
        
        return {
            'age': int(parts[0]),
            'income': float(parts[1]),
            'loan_amount': float(parts[2]),
            'credit_score': int(parts[3]),
            'employment_years': int(parts[4]),
            'debt_to_income': float(parts[5])
        }
    except:
        return None


def format_metric_card(label: str, value: Any, delta: str = None, help_text: str = None):
    """Format a metric card with optional delta and help text"""
    if isinstance(value, float):
        if 0 <= value <= 1:
            value_str = f"{value:.1%}"
        else:
            value_str = f"{value:.3f}"
    else:
        value_str = str(value)
    
    st.metric(label=label, value=value_str, delta=delta, help=help_text)