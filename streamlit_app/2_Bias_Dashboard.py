# Fairness metrics and bias analysis graphs
"""
Bias Dashboard - Fairness Metrics and Analysis
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils_streamlit import call_api, plot_bias_metrics, plot_fairness_gauge, format_metric_card

st.set_page_config(page_title="Bias Dashboard", page_icon="⚖️", layout="wide")

st.title("⚖️ Bias & Fairness Dashboard")
st.markdown("Detect and analyze **demographic bias** in your ML model predictions.")

st.markdown("---")

# Input method
input_method = st.radio(
    "Choose input method:",
    ["Upload CSV", "JSON Input"],
    horizontal=True
)

data = None
sensitive_attr = None
true_label_col = "true_label"

if input_method == "Upload CSV":
    st.markdown("### 📂 Upload Dataset")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file with predictions and sensitive attributes",
        type=['csv'],
        help="CSV must contain feature columns, true_label, and sensitive attribute (e.g., gender, ethnicity)"
    )
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} rows with {len(df.columns)} columns")
            
            with st.expander("📊 Preview Data"):
                st.dataframe(df.head(10), use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                sensitive_attr = st.selectbox(
                    "Select Sensitive Attribute:",
                    options=[col for col in df.columns if col not in ['true_label', 'loan_approved']],
                    help="Column containing demographic groups (e.g., gender, ethnicity)"
                )
            
            with col2:
                true_label_options = [col for col in df.columns if 'label' in col.lower() or 'approved' in col.lower()]
                if not true_label_options:
                    true_label_options = df.columns.tolist()
                
                true_label_col = st.selectbox(
                    "Select True Label Column:",
                    options=true_label_options,
                    help="Column containing actual outcomes (0 or 1)"
                )
            
            # Store file for API call
            uploaded_file.seek(0)
            st.session_state['csv_file'] = uploaded_file
            st.session_state['csv_data'] = df
            
        except Exception as e:
            st.error(f"❌ Error reading CSV: {e}")

else:  # JSON Input
    st.markdown("### 📄 Enter JSON Data")
    
    default_json = """[
    {
        "age": 35, "income": 75000, "loan_amount": 25000,
        "credit_score": 720, "employment_years": 8,
        "debt_to_income": 0.25, "gender": "Male", "true_label": 1
    },
    {
        "age": 28, "income": 45000, "loan_amount": 15000,
        "credit_score": 650, "employment_years": 3,
        "debt_to_income": 0.45, "gender": "Female", "true_label": 0
    }
]"""
    
    json_input = st.text_area(
        "Paste JSON array with samples:",
        value=default_json,
        height=200
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        sensitive_attr = st.text_input("Sensitive Attribute Name:", value="gender")
    
    with col2:
        true_label_col = st.text_input("True Label Column Name:", value="true_label")
    
    try:
        import json
        data = json.loads(json_input)
        st.success(f"✅ Valid JSON with {len(data)} sample(s)")
    except:
        st.error("❌ Invalid JSON format")
        data = None

# Run bias analysis
if st.button("🚀 Run Bias Analysis", type="primary", use_container_width=True):
    if not sensitive_attr:
        st.error("⚠️ Please select a sensitive attribute")
        st.stop()
    
    with st.spinner("🔄 Computing fairness metrics..."):
        
        if input_method == "Upload CSV" and 'csv_file' in st.session_state:
            # CSV upload
            csv_file = st.session_state['csv_file']
            csv_file.seek(0)
            
            files = {'file': (csv_file.name, csv_file, 'text/csv')}
            form_data = {
                'sensitive_attr': sensitive_attr,
                'true_label_col': true_label_col
            }
            
            success, result = call_api("/api/bias/upload", method="POST", files=files, data=form_data)
            
        else:
            # JSON input
            if not data:
                st.error("⚠️ Please provide valid JSON data")
                st.stop()
            
            payload = {
                "data": data,
                "sensitive_attr": sensitive_attr,
                "true_label_col": true_label_col
            }
            
            success, result = call_api("/api/bias", method="POST", data=payload)
        
        if not success:
            st.error(f"❌ API Error: {result.get('error', 'Unknown error')}")
            st.stop()
        
        st.success(f"✅ Analysis complete for {result.get('num_samples', 0)} samples")
        
        # Store in session state
        st.session_state['bias_result'] = result

# Display results
if 'bias_result' in st.session_state:
    result = st.session_state['bias_result']
    
    st.markdown("---")
    st.markdown("## 📊 Fairness Analysis Results")
    
    # Key metrics
    st.markdown("### 🎯 Key Fairness Metrics")
    
    overall = result.get('overall_metrics', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        di = overall.get('disparate_impact', 1.0)
        di_status = "🟢 Pass" if di >= 0.8 else "🔴 Fail"
        format_metric_card(
            "Disparate Impact",
            di,
            delta=di_status,
            help_text="Ratio of positive rates (should be ≥ 0.8)"
        )
    
    with col2:
        dp = overall.get('demographic_parity_difference', 0.0)
        dp_status = "🟢 Pass" if dp <= 0.1 else "🔴 Fail"
        format_metric_card(
            "Demographic Parity",
            dp,
            delta=dp_status,
            help_text="Difference in positive rates (should be ≤ 0.1)"
        )
    
    with col3:
        eo = overall.get('equal_opportunity_difference', 0.0)
        eo_status = "🟢 Pass" if eo <= 0.1 else "🔴 Fail"
        format_metric_card(
            "Equal Opportunity",
            eo,
            delta=eo_status,
            help_text="Difference in TPR (should be ≤ 0.1)"
        )
    
    with col4:
        ap = overall.get('accuracy_parity_difference', 0.0)
        format_metric_card(
            "Accuracy Parity",
            ap,
            help_text="Difference in accuracy across groups"
        )
    
    # Disparate Impact Gauge
    st.markdown("### 📏 Disparate Impact Gauge")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = plot_fairness_gauge(overall.get('disparate_impact', 1.0))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Interpretation")
        di = overall.get('disparate_impact', 1.0)
        
        if di >= 0.8:
            st.success("✅ **No Violation**\n\nModel passes the 80% rule for disparate impact.")
        else:
            st.error(f"❌ **Violation Detected**\n\nDisparate impact ({di:.2f}) is below the 0.8 threshold.")
            st.warning("**Recommended Actions:**\n1. Review training data distribution\n2. Consider reweighting samples\n3. Adjust decision thresholds")
    
    # Group-level metrics
    st.markdown("---")
    st.markdown("### 👥 Performance by Group")
    
    group_metrics = result.get('group_metrics', {})
    
    if group_metrics:
        # Bar chart
        fig = plot_bias_metrics(group_metrics)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        st.markdown("#### Detailed Metrics Table")
        
        rows = []
        for group, metrics in group_metrics.items():
            rows.append({
                'Group': group,
                'Sample Size': metrics.get('group_size', 0),
                'Positive Rate': f"{metrics.get('positive_rate', 0):.1%}",
                'Accuracy': f"{metrics.get('accuracy', 0):.1%}",
                'TPR (Recall)': f"{metrics.get('true_positive_rate', 0):.1%}",
                'TNR': f"{metrics.get('true_negative_rate', 0):.1%}",
                'FPR': f"{metrics.get('false_positive_rate', 0):.1%}",
                'FNR': f"{metrics.get('false_negative_rate', 0):.1%}",
                'Precision': f"{metrics.get('precision', 0):.1%}",
                'F1 Score': f"{metrics.get('f1_score', 0):.3f}"
            })
        
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Warnings
    warnings = result.get('warnings', [])
    if warnings:
        st.markdown("---")
        st.markdown("### ⚠️ Warnings & Alerts")
        
        for warning in warnings:
            if "Disparate Impact" in warning or "Demographic Parity" in warning or "Equal Opportunity" in warning:
                st.error(f"🔴 {warning}")
            else:
                st.warning(f"⚠️ {warning}")
    
    # Export results
    st.markdown("---")
    st.markdown("### 💾 Export Results")
    
    import json
    json_str = json.dumps(result, indent=2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name="bias_analysis_results.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        # Create summary report
        summary = f"""FAIRLENS AI - BIAS ANALYSIS REPORT
Generated: {pd.Timestamp.now()}

Dataset: {result.get('num_samples', 0)} samples
Sensitive Attribute: {result.get('sensitive_attribute', 'N/A')}
Groups: {', '.join(result.get('groups', []))}

OVERALL FAIRNESS METRICS:
- Disparate Impact: {overall.get('disparate_impact', 0):.3f}
- Demographic Parity Difference: {overall.get('demographic_parity_difference', 0):.3f}
- Equal Opportunity Difference: {overall.get('equal_opportunity_difference', 0):.3f}

GROUP PERFORMANCE:
"""
        for group, metrics in group_metrics.items():
            summary += f"\n{group} (n={metrics.get('group_size', 0)}):\n"
            summary += f"  Positive Rate: {metrics.get('positive_rate', 0):.1%}\n"
            summary += f"  Accuracy: {metrics.get('accuracy', 0):.1%}\n"
        
        if warnings:
            summary += "\nWARNINGS:\n"
            for warning in warnings:
                summary += f"- {warning}\n"
        
        st.download_button(
            label="📄 Download Report",
            data=summary,
            file_name="bias_analysis_report.txt",
            mime="text/plain",
            use_container_width=True
        )

else:
    st.info("👆 Upload a dataset or enter JSON data above and click 'Run Bias Analysis'")

# Help section
with st.expander("❓ Help & Understanding Fairness Metrics"):
    st.markdown("""
    ### Fairness Metrics Explained
    
    **Disparate Impact (80% Rule)**
    - Ratio of positive prediction rates between groups
    - Should be ≥ 0.8 (favored group should not be > 1.25x the other)
    - Example: If males get 60% approval and females get 45%, DI = 0.75 (violation)
    
    **Demographic Parity**
    - Difference in positive prediction rates
    - Should be ≤ 0.1 (10% difference)
    - Ensures similar approval rates across groups
    
    **Equal Opportunity**
    - Difference in True Positive Rates (recall)
    - Should be ≤ 0.1
    - Ensures qualified applicants have equal chances
    
    **Accuracy Parity**
    - Difference in accuracy across groups
    - Lower is better
    - Ensures model performs equally well for all groups
    
    ### How to Address Bias
    
    1. **Data Collection**: Ensure balanced representation
    2. **Reweighting**: Give underrepresented groups more weight
    3. **Resampling**: Oversample minority groups or undersample majority
    4. **Threshold Adjustment**: Use different decision thresholds per group
    5. **Feature Engineering**: Remove or add features to reduce bias
    6. **Model Retraining**: Train with fairness constraints
    
    ### Interpreting Results
    
    - 🟢 **Green**: Metric passes fairness threshold
    - 🔴 **Red**: Metric fails threshold (action needed)
    - ⚠️ **Yellow**: Warning (small sample size, monitor closely)
    """)