# Visualization of SHAP, LIME plots

"""
Explainability Page - SHAP and LIME Analysis
"""
import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils_streamlit import (
    call_api, plot_shap_waterfall, plot_shap_bar,
    display_lime_table, parse_sample_input
)

st.set_page_config(page_title="Explainability", page_icon="🔍", layout="wide")

st.title("🔍 Model Explainability")
st.markdown("Understand **why** your model makes certain predictions using SHAP and LIME.")

st.markdown("---")

# Input method selection
input_method = st.radio(
    "Choose input method:",
    ["Manual Input", "JSON Input"],
    horizontal=True
)

sample_data = None

if input_method == "Manual Input":
    st.markdown("### 📝 Enter Sample Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        income = st.number_input("Income ($)", min_value=0, value=75000, step=1000)
    
    with col2:
        loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=25000, step=1000)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=720)
    
    with col3:
        employment_years = st.number_input("Employment Years", min_value=0, max_value=50, value=8)
        debt_to_income = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
    
    sample_data = [{
        "age": age,
        "income": income,
        "loan_amount": loan_amount,
        "credit_score": credit_score,
        "employment_years": employment_years,
        "debt_to_income": debt_to_income
    }]

else:  # JSON Input
    st.markdown("### 📄 Enter JSON Data")
    
    default_json = """[
    {
        "age": 35,
        "income": 75000,
        "loan_amount": 25000,
        "credit_score": 720,
        "employment_years": 8,
        "debt_to_income": 0.25
    }
]"""
    
    json_input = st.text_area(
        "Paste JSON array of samples:",
        value=default_json,
        height=200
    )
    
    try:
        import json
        sample_data = json.loads(json_input)
        st.success(f"✅ Valid JSON with {len(sample_data)} sample(s)")
    except:
        st.error("❌ Invalid JSON format")
        sample_data = None

# Explainer selection
st.markdown("### ⚙️ Configuration")
col1, col2 = st.columns(2)

with col1:
    explainer_type = st.selectbox(
        "Select Explainer:",
        ["both", "shap", "lime"],
        help="SHAP: Fast, tree-based. LIME: Model-agnostic."
    )

with col2:
    st.info(f"""
    **Selected:** {explainer_type.upper()}
    
    - **SHAP**: Global + local explanations
    - **LIME**: Local explanations only
    - **Both**: Complete analysis (slower)
    """)

# Run explanation
if st.button("🚀 Run Explainability Analysis", type="primary", use_container_width=True):
    if not sample_data:
        st.error("⚠️ Please provide valid input data")
        st.stop()
    
    with st.spinner("🔄 Computing explanations... This may take 10-30 seconds"):
        
        payload = {
            "data": sample_data,
            "explainer_type": explainer_type
        }
        
        success, result = call_api("/api/explain", method="POST", data=payload)
        
        if not success:
            st.error(f"❌ API Error: {result.get('error', 'Unknown error')}")
            st.stop()
        
        st.success(f"✅ Analysis complete for {result.get('num_samples', 0)} sample(s)")
        
        # Store in session state
        st.session_state['explain_result'] = result
        st.session_state['sample_data'] = sample_data

# Display results if available
if 'explain_result' in st.session_state:
    result = st.session_state['explain_result']
    
    st.markdown("---")
    st.markdown("## 📊 Results")
    
    # SHAP Results
    if result.get('shap_explanation'):
        st.markdown("### 🎯 SHAP Analysis")
        
        shap_data = result['shap_explanation']
        local_exp = shap_data.get('local_explanations', {})
        global_imp = shap_data.get('global_feature_importance', {})
        
        tab1, tab2 = st.tabs(["📈 Global Importance", "🔬 Local Explanation"])
        
        with tab1:
            st.markdown("#### Feature Importance Across All Predictions")
            
            if global_imp:
                feature_names = global_imp.get('feature_names', [])
                importance_scores = global_imp.get('importance_scores', [])
                
                if feature_names and importance_scores:
                    # Plot
                    fig = plot_shap_bar(feature_names, importance_scores)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Table
                    import pandas as pd
                    df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importance_scores
                    })
                    
                    st.dataframe(
                        df.style.format({'Importance': '{:.4f}'}),
                        use_container_width=True,
                        hide_index=True
                    )
        
        with tab2:
            st.markdown("#### Feature Contributions for Individual Prediction")
            
            if local_exp:
                shap_values = local_exp.get('shap_values', [[]])[0]
                feature_names = local_exp.get('feature_names', [])
                base_value = local_exp.get('base_value', 0)
                
                if shap_values and feature_names:
                    # Waterfall plot
                    fig = plot_shap_waterfall(shap_values, feature_names, base_value)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Explanation
                    st.markdown("#### 💡 Interpretation")
                    
                    import pandas as pd
                    df = pd.DataFrame({
                        'Feature': feature_names,
                        'SHAP Value': shap_values,
                        'Impact': ['🔴 Decreases' if v < 0 else '🟢 Increases' for v in shap_values]
                    })
                    df = df.reindex(df['SHAP Value'].abs().sort_values(ascending=False).index)
                    
                    st.dataframe(
                        df.style.format({'SHAP Value': '{:.4f}'}),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    st.info(f"""
                    **Base Value (Average Prediction):** {base_value:.4f}
                    
                    Features with positive SHAP values push the prediction higher.
                    Features with negative SHAP values push the prediction lower.
                    """)
    
    # LIME Results
    if result.get('lime_explanation'):
        st.markdown("---")
        st.markdown("### 🔬 LIME Analysis")
        
        lime_data = result['lime_explanation']
        explanations = lime_data.get('explanations', [])
        
        if explanations:
            st.markdown("#### Local Feature Contributions")
            
            for i, exp in enumerate(explanations):
                with st.expander(f"Sample {i+1} Explanation", expanded=(i == 0)):
                    
                    # Show prediction probability if available
                    pred_prob = exp.get('prediction_probability')
                    if pred_prob is not None:
                        st.metric("Prediction Probability", f"{pred_prob:.1%}")
                    
                    # Display feature weights
                    display_lime_table([exp])
                    
                    st.caption(f"Explained {exp.get('num_features_explained', 0)} features")
    
    # Download results
    st.markdown("---")
    st.markdown("### 💾 Export Results")
    
    import json
    json_str = json.dumps(result, indent=2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name="explainability_results.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        # Create summary text
        summary = f"""FAIRLENS AI - EXPLAINABILITY REPORT
Generated: {pd.Timestamp.now()}

Samples Analyzed: {result.get('num_samples', 0)}
Explainer Type: {explainer_type.upper()}

Top 3 Most Important Features:
"""
        if global_imp:
            for i, feat in enumerate(global_imp.get('feature_names', [])[:3], 1):
                idx = global_imp.get('feature_names', []).index(feat)
                score = global_imp.get('importance_scores', [])[idx]
                summary += f"{i}. {feat}: {score:.4f}\n"
        
        st.download_button(
            label="📄 Download Summary",
            data=summary,
            file_name="explainability_summary.txt",
            mime="text/plain",
            use_container_width=True
        )

else:
    st.info("👆 Enter sample data above and click 'Run Explainability Analysis' to see results")

# Help section
with st.expander("❓ Help & Tips"):
    st.markdown("""
    ### How to Use This Page
    
    1. **Choose Input Method**: Manual entry or JSON for multiple samples
    2. **Select Explainer**: 
       - SHAP for global + local explanations (faster)
       - LIME for model-agnostic explanations
       - Both for complete analysis
    3. **Run Analysis**: Click the button and wait for results
    4. **Interpret Results**:
       - **SHAP Waterfall**: Shows how each feature contributes
       - **Feature Importance**: Ranking of most influential features
       - **LIME Table**: Local feature contributions with conditions
    
    ### Understanding SHAP Values
    - **Positive values** (green): Feature increases prediction
    - **Negative values** (red): Feature decreases prediction
    - **Magnitude**: Larger absolute values = stronger impact
    
    ### Understanding LIME
    - Shows feature conditions (e.g., "credit_score > 700")
    - Weight indicates strength and direction of impact
    - Model-agnostic (works with any model type)
    
    ### Tips
    - Use SHAP for faster analysis on tree-based models
    - Use LIME for explaining black-box models
    - Analyze multiple samples to find patterns
    """)