# Natural language model explanation using LangChain
"""
LLM Summary Page - Natural Language Explanations
"""
import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils_streamlit import call_api, create_summary_card

st.set_page_config(page_title="LLM Summary", page_icon="🤖", layout="wide")

st.title("🤖 AI-Powered Insights")
st.markdown("Get **natural language summaries** and **actionable recommendations** powered by Google Gemini.")

st.markdown("---")

# Check LLM status
success, status = call_api("/api/summary/status", method="GET")

if success and status.get('llm_available'):
    st.success(f"✅ {status.get('message', 'LLM service ready')}")
    st.caption(f"Framework: {status.get('framework', 'LangChain')} | Model: {status.get('llm_model', 'gemini-pro')}")
else:
    st.error("❌ LLM service not available. Please configure GOOGLE_API_KEY in .env file.")
    st.stop()

st.markdown("---")

# Check if we have previous results
has_explain = 'explain_result' in st.session_state
has_bias = 'bias_result' in st.session_state

if not has_explain and not has_bias:
    st.warning("""
    ⚠️ No analysis results found in session.
    
    Please run analysis first:
    1. Go to **📈 Explainability** page to get SHAP/LIME results
    2. Go to **⚖️ Bias Dashboard** to get fairness metrics
    3. Then return here for AI-powered summaries
    """)
    
    st.info("Or manually input analysis results below:")

# Manual input option
with st.expander("📝 Manual Input (Advanced)"):
    st.markdown("Paste your analysis results as JSON:")
    
    tab1, tab2 = st.tabs(["SHAP/LIME Results", "Bias Results"])
    
    with tab1:
        shap_json = st.text_area(
            "SHAP Explanation JSON:",
            height=200,
            placeholder='{"global_feature_importance": {"feature_names": [...], "importance_scores": [...]}}'
        )
    
    with tab2:
        bias_json = st.text_area(
            "Bias Results JSON:",
            height=200,
            placeholder='{"sensitive_attribute": "gender", "groups": [...], "overall_metrics": {...}}'
        )
    
    if st.button("Load Manual Input"):
        import json
        try:
            if shap_json:
                st.session_state['explain_result'] = {'shap_explanation': json.loads(shap_json)}
            if bias_json:
                st.session_state['bias_result'] = json.loads(bias_json)
            st.success("✅ Manual data loaded")
            st.rerun()
        except json.JSONDecodeError as e:
            st.error(f"❌ Invalid JSON: {e}")

# Summary generation options
st.markdown("### ⚙️ Generate Summary")

summary_type = st.radio(
    "Choose summary type:",
    ["Complete Analysis (Recommended)", "Explainability Only", "Bias Only"],
    horizontal=True
)

# Generate button
if st.button("🚀 Generate AI Summary", type="primary", use_container_width=True):
    
    with st.spinner("🤖 Generating natural language summary with Gemini... This may take 5-15 seconds"):
        
        if summary_type == "Explainability Only":
            if not has_explain:
                st.error("❌ No explainability results found. Run analysis on Explainability page first.")
                st.stop()
            
            explain_data = st.session_state.get('explain_result', {})
            payload = {
                "shap_explanation": explain_data.get('shap_explanation'),
                "lime_explanation": explain_data.get('lime_explanation')
            }
            
            success, result = call_api("/api/summary/explainability-only", method="POST", data=payload)
        
        elif summary_type == "Bias Only":
            if not has_bias:
                st.error("❌ No bias results found. Run analysis on Bias Dashboard page first.")
                st.stop()
            
            bias_data = st.session_state.get('bias_result', {})
            
            success, result = call_api("/api/summary/bias-only", method="POST", data=bias_data)
        
        else:  # Complete Analysis
            if not has_explain and not has_bias:
                st.error("❌ No analysis results found. Run analysis first.")
                st.stop()
            
            payload = {}
            
            if has_explain:
                explain_data = st.session_state.get('explain_result', {})
                payload['shap_explanation'] = explain_data.get('shap_explanation')
                payload['lime_explanation'] = explain_data.get('lime_explanation')
            
            if has_bias:
                bias_data = st.session_state.get('bias_result', {})
                payload['bias_results'] = bias_data
            
            success, result = call_api("/api/summary", method="POST", data=payload)
        
        if not success:
            st.error(f"❌ API Error: {result.get('error', 'Unknown error')}")
            st.stop()
        
        st.success("✅ Summary generated successfully!")
        
        # Store result
        st.session_state['llm_summary'] = result

# Display results
if 'llm_summary' in st.session_state:
    result = st.session_state['llm_summary']
    
    st.markdown("---")
    st.markdown("## 📋 AI-Generated Analysis")
    
    # Explainability Summary
    if result.get('explainability_summary') or result.get('summary'):
        exp_summary = result.get('explainability_summary') or result.get('summary')
        
        if 'summary' in result and not result.get('explainability_summary'):
            # Single summary from explainability-only or bias-only
            if summary_type == "Explainability Only":
                create_summary_card("Model Explainability Insights", exp_summary, "🔍")
            elif summary_type == "Bias Only":
                st.markdown("### ⚖️ Fairness Analysis")
                
                # Try to parse structured bias summary
                if "EXECUTIVE SUMMARY:" in exp_summary and "RECOMMENDATIONS:" in exp_summary:
                    parts = exp_summary.split("RECOMMENDATIONS:")
                    exec_summary = parts[0].replace("EXECUTIVE SUMMARY:", "").strip()
                    recommendations = parts[1].strip() if len(parts) > 1 else ""
                    
                    create_summary_card("Executive Summary", exec_summary, "📊")
                    
                    if recommendations:
                        st.markdown("### 💡 Recommendations")
                        st.markdown(recommendations)
                else:
                    create_summary_card("Bias Analysis", exp_summary, "⚖️")
        else:
            create_summary_card("Model Explainability Insights", exp_summary, "🔍")
    
    # Bias Summary
    if result.get('bias_summary'):
        bias_summary = result.get('bias_summary')
        
        st.markdown("---")
        st.markdown("### ⚖️ Fairness & Bias Analysis")
        
        # Try to parse structured summary
        if "EXECUTIVE SUMMARY:" in bias_summary and "RECOMMENDATIONS:" in bias_summary:
            parts = bias_summary.split("RECOMMENDATIONS:")
            exec_summary = parts[0].replace("EXECUTIVE SUMMARY:", "").strip()
            recommendations = parts[1].strip() if len(parts) > 1 else ""
            
            create_summary_card("Executive Summary", exec_summary, "📊")
            
            if recommendations:
                st.markdown("### 💡 Fairness Recommendations")
                st.markdown(recommendations)
        else:
            create_summary_card("Fairness Analysis", bias_summary, "⚖️")
    
    # Combined Recommendations
    if result.get('recommendations'):
        st.markdown("---")
        st.markdown("### 🎯 Strategic Recommendations")
        
        recommendations = result.get('recommendations')
        st.markdown(recommendations)
    
    # Export
    st.markdown("---")
    st.markdown("### 💾 Export Summary")
    
    import json
    json_str = json.dumps(result, indent=2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name="llm_summary.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        # Create text summary
        text_summary = "FAIRLENS AI - AI-POWERED ANALYSIS REPORT\n"
        text_summary += "=" * 60 + "\n\n"
        
        if result.get('explainability_summary'):
            text_summary += "EXPLAINABILITY INSIGHTS:\n"
            text_summary += "-" * 60 + "\n"
            text_summary += result.get('explainability_summary') + "\n\n"
        
        if result.get('bias_summary'):
            text_summary += "FAIRNESS ANALYSIS:\n"
            text_summary += "-" * 60 + "\n"
            text_summary += result.get('bias_summary') + "\n\n"
        
        if result.get('recommendations'):
            text_summary += "STRATEGIC RECOMMENDATIONS:\n"
            text_summary += "-" * 60 + "\n"
            text_summary += result.get('recommendations') + "\n"
        
        st.download_button(
            label="📄 Download Text Report",
            data=text_summary,
            file_name="fairlens_analysis_report.txt",
            mime="text/plain",
            use_container_width=True
        )

else:
    st.info("👆 Select a summary type and click 'Generate AI Summary' to see results")

# Example output
with st.expander("💡 Example Summary Output"):
    st.markdown("""
    ### What to Expect
    
    **Explainability Summary:**
    > "The model's predictions are primarily driven by credit_score (importance: 0.45), 
    > followed by income (0.32) and loan_amount (0.23). This indicates that an applicant's 
    > creditworthiness is the most critical factor in approval decisions, with financial 
    > capacity playing a supporting role."
    
    **Bias Analysis (Executive Summary):**
    > "The model exhibits significant fairness concerns with a disparate impact of 0.67, 
    > falling below the 0.8 threshold. Female applicants receive positive decisions at 40% 
    > compared to 60% for males, indicating high-severity bias requiring immediate attention."
    
    **Recommendations:**
    > 1. **Collect balanced training data** - Expand data collection to achieve 50/50 
    >    gender representation with at least 500 samples per group
    > 2. **Implement sample reweighting** - Apply inverse class weights (1.5x for females) 
    >    during model training
    > 3. **Adjust decision thresholds** - Evaluate group-specific thresholds to equalize 
    >    TPR while maintaining accuracy within 2%
    
    All summaries are generated by Google Gemini and are based strictly on your analysis data.
    """)