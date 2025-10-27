"""
Summary Service using LangChain + Google Gemini
This properly implements LangChain integration as per Phase 3 requirements
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import Dict, Any, List, Optional
import json

from app.utils.logger import get_logger
from app.config import settings

logger = get_logger()


class SummaryService:
    """LangChain-based summary service using Gemini"""
    
    def __init__(self):
        """Initialize LangChain with Gemini"""
        self.llm = None
        self._initialize_langchain()
    
    def _initialize_langchain(self):
        """Initialize LangChain with Google Gemini"""
        try:
            if not settings.google_api_key:
                logger.warning("Google API key not found. LLM summaries will not be available.")
                return
            
            # Initialize ChatGoogleGenerativeAI (LangChain wrapper for Gemini)
            self.llm = ChatGoogleGenerativeAI(
                model=settings.llm_model,
                google_api_key=settings.google_api_key,
                temperature=settings.llm_temperature,
                max_output_tokens=settings.llm_max_tokens,
                convert_system_message_to_human=True
            )
            
            logger.success(f"LangChain initialized with Gemini model: {settings.llm_model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LangChain: {e}")
            self.llm = None
    
    def is_available(self) -> bool:
        """Check if LLM service is available"""
        return self.llm is not None
    
    def generate_bias_summary(self, bias_data: Dict[str, Any]) -> str:
        """
        Generate bias analysis summary using LangChain
        
        Args:
            bias_data: Bias detection results
            
        Returns:
            Natural language summary
        """
        try:
            if not self.is_available():
                return "LLM service not available. Please configure GOOGLE_API_KEY."
            
            # Create prompt template
            prompt_template = """You are an AI fairness analyst. Your task is to analyze model fairness metrics and provide actionable recommendations.

**CRITICAL RULES:**
- Base your analysis ONLY on the metrics provided below
- Do NOT make up or invent any claims
- Provide concrete, actionable recommendations
- Be clear about the severity of any issues

**Fairness Analysis Data:**

Sensitive Attribute: {sensitive_attr}
Groups Analyzed: {groups}

**Overall Fairness Metrics:**
- Disparate Impact: {disparate_impact:.3f} {di_status}
- Demographic Parity Difference: {demographic_parity:.3f} {dp_status}
- Equal Opportunity Difference: {equal_opportunity:.3f} {eo_status}

**Group-Level Performance:**
{group_details}

**Your Task:**

1. EXECUTIVE SUMMARY (3-4 sentences):
   - Overall fairness assessment
   - Which groups are most affected
   - Severity level (low/medium/high concern)

2. TOP 3 ACTIONABLE RECOMMENDATIONS (prioritized by impact):
   Provide specific steps such as:
   - Data collection strategies
   - Reweighting or resampling techniques
   - Feature engineering improvements
   - Threshold adjustments
   - Model retraining approaches

Format your response as:

EXECUTIVE SUMMARY:
[Your 3-4 sentence summary]

RECOMMENDATIONS:
1. [Most important action with specific steps]
2. [Second priority with specific steps]
3. [Third priority with specific steps]
"""
            
            # Extract metrics
            sensitive_attr = bias_data.get("sensitive_attribute", "unknown")
            groups = bias_data.get("groups", [])
            overall = bias_data.get("overall_metrics", {})
            
            disparate_impact = overall.get("disparate_impact", 1.0)
            demographic_parity = overall.get("demographic_parity_difference", 0.0)
            equal_opportunity = overall.get("equal_opportunity_difference", 0.0)
            
            # Status indicators
            di_status = "⚠️ VIOLATION (< 0.8)" if disparate_impact < 0.8 else "✅ OK"
            dp_status = "⚠️ VIOLATION (> 0.1)" if demographic_parity > 0.1 else "✅ OK"
            eo_status = "⚠️ VIOLATION (> 0.1)" if equal_opportunity > 0.1 else "✅ OK"
            
            # Format group details
            group_metrics = bias_data.get("group_metrics", {})
            group_details = ""
            for group, metrics in group_metrics.items():
                group_details += f"\n{group} (n={metrics.get('group_size', 0)}):\n"
                group_details += f"  - Positive Rate: {metrics.get('positive_rate', 0):.1%}\n"
                group_details += f"  - Accuracy: {metrics.get('accuracy', 0):.1%}\n"
                group_details += f"  - True Positive Rate: {metrics.get('true_positive_rate', 0):.1%}\n"
            
            # Create LangChain prompt
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=[
                    "sensitive_attr", "groups", "disparate_impact", "di_status",
                    "demographic_parity", "dp_status", "equal_opportunity", "eo_status",
                    "group_details"
                ]
            )
            
            # Create LangChain chain
            chain = LLMChain(llm=self.llm, prompt=prompt)
            
            # Generate summary
            logger.info("Generating bias summary with LangChain")
            result = chain.run(
                sensitive_attr=sensitive_attr,
                groups=", ".join(map(str, groups)),
                disparate_impact=disparate_impact,
                di_status=di_status,
                demographic_parity=demographic_parity,
                dp_status=dp_status,
                equal_opportunity=equal_opportunity,
                eo_status=eo_status,
                group_details=group_details
            )
            
            logger.success("Bias summary generated successfully")
            return result
            
        except Exception as e:
            logger.error(f"Bias summary generation failed: {e}")
            return f"Error generating bias summary: {str(e)}"
    
    def generate_explainability_summary(
        self,
        shap_data: Optional[Dict[str, Any]] = None,
        lime_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate explainability summary using LangChain
        
        Args:
            shap_data: SHAP explanation results
            lime_data: LIME explanation results
            
        Returns:
            Natural language summary
        """
        try:
            if not self.is_available():
                return "LLM service not available. Please configure GOOGLE_API_KEY."
            
            # Create prompt template
            prompt_template = """You are an ML explainability expert. Explain model predictions in simple, non-technical language.

**IMPORTANT RULES:**
- Base your analysis ONLY on the data provided
- Do NOT make up information
- Keep explanations clear and concise (3-4 sentences)
- Avoid technical jargon
- Focus on practical insights

**Model Explainability Data:**

{feature_importance}

**Your Task:**

Provide a brief explanation (3-4 sentences) that:
1. Identifies the most important features driving predictions
2. Explains what this means in practical terms
3. Highlights any notable patterns or insights

Write in plain English for non-technical stakeholders.
"""
            
            # Build feature importance section
            feature_importance = ""
            
            if shap_data and "global_feature_importance" in shap_data:
                features = shap_data["global_feature_importance"].get("feature_names", [])
                scores = shap_data["global_feature_importance"].get("importance_scores", [])
                
                feature_importance += "**Top Features by Importance (SHAP):**\n"
                for feat, score in zip(features[:5], scores[:5]):
                    feature_importance += f"- {feat}: {score:.4f}\n"
            
            if lime_data and "explanations" in lime_data:
                feature_importance += "\n**Feature Contributions (LIME):**\n"
                first_exp = lime_data["explanations"][0] if lime_data["explanations"] else None
                if first_exp and "feature_weights" in first_exp:
                    for fw in first_exp["feature_weights"][:5]:
                        feature_importance += f"- {fw['feature']}: {fw['weight']:.4f}\n"
            
            # Create LangChain prompt
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["feature_importance"]
            )
            
            # Create chain
            chain = LLMChain(llm=self.llm, prompt=prompt)
            
            # Generate summary
            logger.info("Generating explainability summary with LangChain")
            result = chain.run(feature_importance=feature_importance)
            
            logger.success("Explainability summary generated")
            return result
            
        except Exception as e:
            logger.error(f"Explainability summary generation failed: {e}")
            return f"Error generating explainability summary: {str(e)}"
    
    def generate_combined_recommendations(
        self,
        shap_data: Optional[Dict[str, Any]],
        bias_data: Dict[str, Any]
    ) -> str:
        """
        Generate combined recommendations using LangChain
        
        Args:
            shap_data: SHAP explanation results
            bias_data: Bias detection results
            
        Returns:
            Combined recommendations
        """
        try:
            if not self.is_available():
                return "LLM service not available."
            
            prompt_template = """You are a senior ML engineer and fairness expert. Provide strategic recommendations based on both explainability and fairness analysis.

**Model Insights:**

Most Influential Features: {top_features}

Fairness Concerns:
- Sensitive Attribute: {sensitive_attr}
- Disparate Impact: {disparate_impact:.3f}
- Groups: {groups}

**Your Task:**

Provide 3 prioritized, actionable recommendations that address both explainability insights and fairness concerns.

Format each as:
[Action] - [Why it matters] - [How to implement]

Be specific and practical.
"""
            
            # Extract data
            top_features = "Unknown"
            if shap_data and "global_feature_importance" in shap_data:
                features = shap_data["global_feature_importance"].get("feature_names", [])
                top_features = ", ".join(features[:3])
            
            sensitive_attr = bias_data.get("sensitive_attribute", "unknown")
            disparate_impact = bias_data.get("overall_metrics", {}).get("disparate_impact", 1.0)
            groups = ", ".join(map(str, bias_data.get("groups", [])))
            
            # Create chain
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["top_features", "sensitive_attr", "disparate_impact", "groups"]
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            
            # Generate
            logger.info("Generating combined recommendations with LangChain")
            result = chain.run(
                top_features=top_features,
                sensitive_attr=sensitive_attr,
                disparate_impact=disparate_impact,
                groups=groups
            )
            
            logger.success("Combined recommendations generated")
            return result
            
        except Exception as e:
            logger.error(f"Combined recommendations failed: {e}")
            return f"Error generating recommendations: {str(e)}"
    
    def generate_complete_summary(
        self,
        shap_data: Optional[Dict[str, Any]] = None,
        lime_data: Optional[Dict[str, Any]] = None,
        bias_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Generate all summaries
        
        Returns:
            Dictionary with all summary sections
        """
        summaries = {}
        
        try:
            # Explainability summary
            if shap_data or lime_data:
                summaries["explainability_summary"] = self.generate_explainability_summary(
                    shap_data, lime_data
                )
            
            # Bias summary
            if bias_data:
                summaries["bias_summary"] = self.generate_bias_summary(bias_data)
            
            # Combined recommendations
            if (shap_data or lime_data) and bias_data:
                summaries["recommendations"] = self.generate_combined_recommendations(
                    shap_data, bias_data
                )
            
            return summaries
            
        except Exception as e:
            logger.error(f"Complete summary generation failed: {e}")
            return {"error": str(e)}


# Global instance
summary_service = SummaryService()