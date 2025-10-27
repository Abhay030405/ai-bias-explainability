# Main landing page for Streamlit app
"""
FairLens AI - Home Page
Streamlit Dashboard for ML Explainability and Bias Detection
"""
import streamlit as st
import requests

# Page configuration
st.set_page_config(
    page_title="FairLens AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .status-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .status-healthy {
        background-color: #d4edda;
        color: #155724;
    }
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">⚖️ FairLens AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Shedding Light on Bias and Transparency in AI Models</div>', unsafe_allow_html=True)

# API Connection Check
API_URL = "http://127.0.0.1:8000"

def check_api_health():
    """Check if FastAPI backend is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except:
        return False, None

api_healthy, health_data = check_api_health()

# Status Banner
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if api_healthy:
        st.markdown(
            '<div class="status-badge status-healthy">✅ API Connected</div>',
            unsafe_allow_html=True
        )
        if health_data:
            st.caption(f"Service: {health_data.get('service', 'FairLens AI')} | Version: {health_data.get('version', '1.0.0')}")
    else:
        st.markdown(
            '<div class="status-badge status-error">❌ API Disconnected</div>',
            unsafe_allow_html=True
        )
        st.error("⚠️ Cannot connect to FastAPI backend. Make sure the server is running on http://127.0.0.1:8000")
        st.code("python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
        st.stop()

st.markdown("---")

# Introduction
st.markdown("## 👋 Welcome to FairLens AI")
st.write("""
FairLens AI is an end-to-end system that makes Machine Learning models **transparent** and **trustworthy**.
Discover why your model makes certain predictions and detect potential biases before they cause harm.
""")

# Features
st.markdown("## ✨ Features")

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="feature-box">', unsafe_allow_html=True)
    st.markdown("### 🔍 **Explainability**")
    st.write("""
    - **SHAP Analysis**: Understand feature contributions to predictions
    - **LIME Explanations**: Local interpretable model-agnostic explanations
    - **Feature Importance**: Identify which features drive decisions
    - **Visual Plots**: Waterfall charts, force plots, and summary visualizations
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="feature-box">', unsafe_allow_html=True)
    st.markdown("### 🤖 **LLM Summaries**")
    st.write("""
    - **Natural Language**: Plain English explanations powered by Google Gemini
    - **Executive Summaries**: Quick overview for stakeholders
    - **Actionable Recommendations**: 3 prioritized steps to improve fairness
    - **Combined Insights**: Holistic analysis of explainability + bias
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="feature-box">', unsafe_allow_html=True)
    st.markdown("### ⚖️ **Bias Detection**")
    st.write("""
    - **Fairness Metrics**: Demographic parity, equal opportunity, disparate impact
    - **Group Analysis**: Compare model performance across sensitive attributes
    - **Violation Detection**: Automatic warnings for bias thresholds
    - **CSV Upload**: Analyze datasets directly from files
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="feature-box">', unsafe_allow_html=True)
    st.markdown("### 📊 **Report Export**")
    st.write("""
    - **JSON Export**: Complete analysis results in structured format
    - **PDF Reports**: Professional documents for compliance teams
    - **Audit Trail**: Timestamped records for regulatory requirements
    - **Shareable**: Easy distribution to stakeholders
    """)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# Quick Start Guide
st.markdown("## 🚀 Quick Start Guide")

st.markdown("""
1. **Navigate** using the sidebar to access different features
2. **Upload** your model predictions or dataset
3. **Analyze** with SHAP, LIME, or bias detection
4. **Review** natural language summaries and recommendations
5. **Export** reports for documentation and compliance

### 📖 Page Navigation:
- **📈 Explainability**: Analyze SHAP and LIME explanations
- **⚖️ Bias Dashboard**: Compute and visualize fairness metrics
- **🧠 LLM Summary**: Get AI-powered insights and recommendations
- **🗂️ Report Export**: Download comprehensive analysis reports
""")

st.markdown("---")

# System Information
with st.expander("🔧 System Information"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Backend API")
        if health_data:
            st.json({
                "status": health_data.get("status"),
                "service": health_data.get("service"),
                "version": health_data.get("version"),
                "model_status": health_data.get("model_status")
            })
    
    with col2:
        st.markdown("### Available Endpoints")
        st.code("""
        GET  /health
        GET  /api/explain/features
        POST /api/explain
        POST /api/bias
        POST /api/bias/upload
        GET  /api/bias/thresholds
        POST /api/summary
        GET  /api/summary/status
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Built with ❤️ using FastAPI, Streamlit, SHAP, LIME, and Google Gemini</p>
    <p>FairLens AI | Version 1.0.0 | Making AI Fair and Transparent</p>
</div>
""", unsafe_allow_html=True)