"""
Report Export Page - Download Complete Analysis
"""
import streamlit as st
import pandas as pd
import json
from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(page_title="Report Export", page_icon="📊", layout="wide")

st.title("📊 Report Export")
st.markdown("Download comprehensive analysis reports in **JSON** or **Text** format.")

st.markdown("---")

# Check what results are available
has_explain = 'explain_result' in st.session_state
has_bias = 'bias_result' in st.session_state
has_summary = 'llm_summary' in st.session_state

st.markdown("### 📦 Available Results")

col1, col2, col3 = st.columns(3)

with col1:
    if has_explain:
        st.success("✅ Explainability Analysis")
    else:
        st.error("❌ No Explainability Data")

with col2:
    if has_bias:
        st.success("✅ Bias Analysis")
    else:
        st.error("❌ No Bias Data")

with col3:
    if has_summary:
        st.success("✅ LLM Summary")
    else:
        st.error("❌ No LLM Summary")

if not (has_explain or has_bias or has_summary):
    st.warning("""
    ⚠️ No analysis results available to export.
    
    Please run analysis first:
    1. **📈 Explainability**: Get SHAP/LIME explanations
    2. **⚖️ Bias Dashboard**: Compute fairness metrics
    3. **🤖 LLM Summary**: Generate AI insights
    
    Then return here to export complete reports.
    """)
    st.stop()

st.markdown("---")

# Report options
st.markdown("### ⚙️ Report Configuration")

col1, col2 = st.columns(2)

with col1:
    report_sections = st.multiselect(
        "Select sections to include:",
        options=["Explainability", "Bias Analysis", "LLM Summary"],
        default=[s for s in ["Explainability", "Bias Analysis", "LLM Summary"] 
                if (s == "Explainability" and has_explain) or 
                   (s == "Bias Analysis" and has_bias) or 
                   (s == "LLM Summary" and has_summary)]
    )

with col2:
    report_format = st.radio(
        "Select format:",
        ["Complete JSON", "Text Report", "Executive Summary"],
        help="JSON: Full data | Text: Readable format | Executive: Key insights only"
    )

# Generate timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Build complete report
def build_complete_report():
    """Build complete report dictionary"""
    report = {
        "report_metadata": {
            "generated_at": datetime.now().isoformat(),
            "fairlens_version": "1.0.0",
            "sections_included": report_sections
        }
    }
    
    if "Explainability" in report_sections and has_explain:
        report["explainability_analysis"] = st.session_state['explain_result']
    
    if "Bias Analysis" in report_sections and has_bias:
        report["bias_analysis"] = st.session_state['bias_result']
    
    if "LLM Summary" in report_sections and has_summary:
        report["llm_summary"] = st.session_state['llm_summary']
    
    return report


def build_text_report():
    """Build human-readable text report"""
    report = f"""
{'=' * 80}
                        FAIRLENS AI ANALYSIS REPORT
{'=' * 80}

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Report ID: {timestamp}

{'=' * 80}
"""
    
    # Explainability Section
    if "Explainability" in report_sections and has_explain:
        explain_data = st.session_state['explain_result']
        report += f"""
SECTION 1: MODEL EXPLAINABILITY
{'-' * 80}

Samples Analyzed: {explain_data.get('num_samples', 0)}
Explainer Type: {explain_data.get('shap_explanation', {}).get('explainer_type', 'N/A')}

"""
        
        # SHAP Global Importance
        if explain_data.get('shap_explanation'):
            shap_data = explain_data['shap_explanation']
            global_imp = shap_data.get('global_feature_importance', {})
            
            if global_imp:
                report += "Top Features by Importance (SHAP):\n"
                for i, (feat, score) in enumerate(zip(
                    global_imp.get('feature_names', []),
                    global_imp.get('importance_scores', [])
                ), 1):
                    report += f"{i}. {feat}: {score:.4f}\n"
                report += "\n"
        
        # LIME Summary
        if explain_data.get('lime_explanation'):
            lime_data = explain_data['lime_explanation']
            report += f"LIME Explanations: {lime_data.get('num_instances', 0)} instance(s)\n\n"
    
    # Bias Section
    if "Bias Analysis" in report_sections and has_bias:
        bias_data = st.session_state['bias_result']
        report += f"""
SECTION 2: FAIRNESS & BIAS ANALYSIS
{'-' * 80}

Dataset: {bias_data.get('num_samples', 0)} samples
Sensitive Attribute: {bias_data.get('sensitive_attribute', 'N/A')}
Groups: {', '.join(bias_data.get('groups', []))}

Overall Fairness Metrics:
"""
        
        overall = bias_data.get('overall_metrics', {})
        report += f"  - Disparate Impact: {overall.get('disparate_impact', 0):.3f}"
        if overall.get('disparate_impact', 1) < 0.8:
            report += " ⚠️ VIOLATION"
        report += "\n"
        
        report += f"  - Demographic Parity Diff: {overall.get('demographic_parity_difference', 0):.3f}\n"
        report += f"  - Equal Opportunity Diff: {overall.get('equal_opportunity_difference', 0):.3f}\n"
        report += f"  - Accuracy Parity Diff: {overall.get('accuracy_parity_difference', 0):.3f}\n\n"
        
        # Group Performance
        report += "Performance by Group:\n"
        for group, metrics in bias_data.get('group_metrics', {}).items():
            report += f"\n{group} (n={metrics.get('group_size', 0)}):\n"
            report += f"  Positive Rate: {metrics.get('positive_rate', 0):.1%}\n"
            report += f"  Accuracy: {metrics.get('accuracy', 0):.1%}\n"
            report += f"  TPR: {metrics.get('true_positive_rate', 0):.1%}\n"
            report += f"  Precision: {metrics.get('precision', 0):.1%}\n"
            report += f"  F1 Score: {metrics.get('f1_score', 0):.3f}\n"
        
        # Warnings
        warnings = bias_data.get('warnings', [])
        if warnings:
            report += f"\nWarnings ({len(warnings)}):\n"
            for warning in warnings:
                report += f"  • {warning}\n"
        report += "\n"
    
    # LLM Summary Section
    if "LLM Summary" in report_sections and has_summary:
        summary_data = st.session_state['llm_summary']
        report += f"""
SECTION 3: AI-POWERED INSIGHTS
{'-' * 80}

"""
        
        if summary_data.get('explainability_summary'):
            report += "Explainability Insights:\n"
            report += summary_data['explainability_summary'] + "\n\n"
        
        if summary_data.get('bias_summary'):
            report += "Fairness Analysis:\n"
            report += summary_data['bias_summary'] + "\n\n"
        
        if summary_data.get('recommendations'):
            report += "Strategic Recommendations:\n"
            report += summary_data['recommendations'] + "\n\n"
    
    report += f"\n{'=' * 80}\n"
    report += "End of Report\n"
    report += f"{'=' * 80}\n"
    
    return report


def build_executive_summary():
    """Build executive summary for stakeholders"""
    summary = f"""
{'=' * 80}
                    FAIRLENS AI - EXECUTIVE SUMMARY
{'=' * 80}

Report Date: {datetime.now().strftime("%Y-%m-%d")}
Document ID: {timestamp}

{'=' * 80}

OVERVIEW
{'-' * 80}

This report provides a high-level summary of ML model fairness and explainability
analysis conducted using FairLens AI.

"""
    
    # Key Findings - Explainability
    if "Explainability" in report_sections and has_explain:
        explain_data = st.session_state['explain_result']
        
        summary += "KEY FINDINGS - MODEL EXPLAINABILITY\n"
        summary += "-" * 80 + "\n\n"
        
        if explain_data.get('shap_explanation'):
            global_imp = explain_data['shap_explanation'].get('global_feature_importance', {})
            features = global_imp.get('feature_names', [])[:3]
            
            summary += f"Top 3 Most Influential Features: {', '.join(features)}\n\n"
        
        if has_summary and st.session_state['llm_summary'].get('explainability_summary'):
            summary += "AI Insights:\n"
            exp_text = st.session_state['llm_summary']['explainability_summary']
            summary += exp_text[:300] + "...\n\n"
    
    # Key Findings - Fairness
    if "Bias Analysis" in report_sections and has_bias:
        bias_data = st.session_state['bias_result']
        
        summary += "KEY FINDINGS - FAIRNESS ASSESSMENT\n"
        summary += "-" * 80 + "\n\n"
        
        overall = bias_data.get('overall_metrics', {})
        di = overall.get('disparate_impact', 1.0)
        
        if di < 0.8:
            summary += f"⚠️ CRITICAL: Disparate Impact Violation Detected ({di:.2f})\n"
            summary += "The model shows significant bias requiring immediate remediation.\n\n"
        else:
            summary += f"✓ Disparate Impact: {di:.2f} (Passes 80% rule)\n\n"
        
        summary += f"Sensitive Attribute: {bias_data.get('sensitive_attribute', 'N/A')}\n"
        summary += f"Groups Analyzed: {', '.join(bias_data.get('groups', []))}\n\n"
        
        # Quick metrics
        summary += "Fairness Metrics:\n"
        summary += f"  • Disparate Impact: {di:.3f}\n"
        summary += f"  • Demographic Parity: {overall.get('demographic_parity_difference', 0):.3f}\n"
        summary += f"  • Equal Opportunity: {overall.get('equal_opportunity_difference', 0):.3f}\n\n"
    
    # Recommendations
    if "LLM Summary" in report_sections and has_summary:
        summary_data = st.session_state['llm_summary']
        
        summary += "PRIORITY RECOMMENDATIONS\n"
        summary += "-" * 80 + "\n\n"
        
        if summary_data.get('recommendations'):
            recs = summary_data['recommendations']
            summary += recs + "\n"
        elif summary_data.get('bias_summary') and "RECOMMENDATIONS:" in summary_data['bias_summary']:
            recs_part = summary_data['bias_summary'].split("RECOMMENDATIONS:")[1]
            summary += recs_part.strip() + "\n"
    
    summary += f"\n{'=' * 80}\n"
    summary += "For detailed analysis, refer to the complete report.\n"
    summary += f"{'=' * 80}\n"
    
    return summary


# Preview
st.markdown("---")
st.markdown("### 👁️ Report Preview")

if report_format == "Complete JSON":
    report_data = build_complete_report()
    st.json(report_data)
elif report_format == "Text Report":
    report_text = build_text_report()
    st.text_area("Report Preview", report_text, height=400)
else:  # Executive Summary
    exec_summary = build_executive_summary()
    st.text_area("Executive Summary Preview", exec_summary, height=400)

# Download buttons
st.markdown("---")
st.markdown("### 💾 Download Report")

col1, col2, col3 = st.columns(3)

with col1:
    if report_format == "Complete JSON":
        report_data = build_complete_report()
        json_str = json.dumps(report_data, indent=2)
        
        st.download_button(
            label="📥 Download JSON Report",
            data=json_str,
            file_name=f"fairlens_complete_report_{timestamp}.json",
            mime="application/json",
            use_container_width=True
        )
    else:
        report_data = build_complete_report()
        json_str = json.dumps(report_data, indent=2)
        
        st.download_button(
            label="📥 Download as JSON",
            data=json_str,
            file_name=f"fairlens_report_{timestamp}.json",
            mime="application/json",
            use_container_width=True
        )

with col2:
    if report_format == "Text Report":
        report_text = build_text_report()
        
        st.download_button(
            label="📄 Download Text Report",
            data=report_text,
            file_name=f"fairlens_text_report_{timestamp}.txt",
            mime="text/plain",
            use_container_width=True
        )
    else:
        report_text = build_text_report()
        
        st.download_button(
            label="📄 Download as Text",
            data=report_text,
            file_name=f"fairlens_report_{timestamp}.txt",
            mime="text/plain",
            use_container_width=True
        )

with col3:
    exec_summary = build_executive_summary()
    
    st.download_button(
        label="📊 Download Executive Summary",
        data=exec_summary,
        file_name=f"fairlens_executive_summary_{timestamp}.txt",
        mime="text/plain",
        use_container_width=True
    )

# Report statistics
st.markdown("---")
st.markdown("### 📈 Report Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_sections = len([s for s in ["Explainability", "Bias Analysis", "LLM Summary"] 
                         if (s == "Explainability" and has_explain) or 
                            (s == "Bias Analysis" and has_bias) or 
                            (s == "LLM Summary" and has_summary)])
    st.metric("Sections Available", total_sections)

with col2:
    selected_sections = len(report_sections)
    st.metric("Sections Selected", selected_sections)

with col3:
    if has_explain:
        samples = st.session_state['explain_result'].get('num_samples', 0)
    elif has_bias:
        samples = st.session_state['bias_result'].get('num_samples', 0)
    else:
        samples = 0
    st.metric("Samples Analyzed", samples)

with col4:
    if report_format == "Complete JSON":
        report_data = build_complete_report()
        size = len(json.dumps(report_data))
    elif report_format == "Text Report":
        size = len(build_text_report())
    else:
        size = len(build_executive_summary())
    
    st.metric("Report Size", f"{size:,} bytes")

# Help
with st.expander("❓ Report Formats Explained"):
    st.markdown("""
    ### Report Format Options
    
    **Complete JSON**
    - Full analysis data in structured JSON format
    - Includes all raw metrics, arrays, and metadata
    - Best for: Programmatic processing, integration with other systems
    - Use when: You need complete data for further analysis
    
    **Text Report**
    - Human-readable format with all sections
    - Includes metrics tables and explanations
    - Best for: Documentation, sharing with technical team
    - Use when: You want detailed readable analysis
    
    **Executive Summary**
    - High-level overview for stakeholders
    - Key findings and priority recommendations only
    - Best for: Management, compliance teams, presentations
    - Use when: You need quick insights without technical details
    
    ### What's Included
    
    Each report contains the sections you select:
    - **Explainability**: SHAP/LIME analysis and feature importance
    - **Bias Analysis**: Fairness metrics and group comparisons
    - **LLM Summary**: AI-generated insights and recommendations
    
    All reports include:
    - Timestamp and report ID for audit trail
    - Metadata about analysis configuration
    - Data quality indicators and warnings
    """)