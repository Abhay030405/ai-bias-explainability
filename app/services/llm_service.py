"""
LLM Service using Google Gemini for natural language summaries
"""
import google.generativeai as genai
from typing import Dict, Any, List, Optional
from app.utils.logger import get_logger
from app.config import settings

logger = get_logger()

class GeminiLLMService:
    """Google Gemini LLM service for generating explanations"""
    
    def __init__(self):
        """Initialize Gemini API"""
        self.model = None
        self._initialize_gemini()
    
    def _initialize_gemini(self):
        """Configure and initialize Gemini model"""
        try:
            if not settings.google_api_key:
                logger.warning("Google API key not found. LLM summaries will not be available.")
                return
            
            # Configure Gemini
            genai.configure(api_key=settings.google_api_key)
            
            # Initialize model
            self.model = genai.GenerativeModel(
                model_name=settings.llm_model,
                generation_config={
                    "temperature": settings.llm_temperature,
                    "max_output_tokens": settings.llm_max_tokens,
                    "top_p": 0.95,
                    "top_k": 40
                }
            )
            
            logger.success(f"Gemini model '{settings.llm_model}' initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            self.model = None
    
    def is_available(self) -> bool:
        """Check if LLM service is available"""
        return self.model is not None
    
    def generate_summary(self, prompt: str) -> str:
        """
        Generate text using Gemini
        
        Args:
            prompt: Input prompt for generation
            
        Returns:
            Generated text response
        """
        try:
            if not self.is_available():
                logger.error("Gemini model not initialized")
                return "LLM service not available. Please configure GOOGLE_API_KEY in .env file."
            
            logger.info("Generating summary with Gemini")
            
            # Generate response
            response = self.model.generate_content(prompt)
            
            # Extract text
            generated_text = response.text
            
            logger.success(f"Generated summary ({len(generated_text)} chars)")
            return generated_text
            
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            return f"Error generating summary: {str(e)}"
    
    def generate_combined_summary(
        self,
        shap_data: Optional[Dict[str, Any]] = None,
        lime_data: Optional[Dict[str, Any]] = None,
        bias_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Generate comprehensive summary from SHAP, LIME, and bias data
        
        Args:
            shap_data: SHAP explanation results
            lime_data: LIME explanation results
            bias_data: Bias detection results
            
        Returns:
            Dictionary with summary sections
        """
        try:
            summaries = {}
            
            # Generate explainability summary
            if shap_data or lime_data:
                logger.info("Generating explainability summary")
                exp_prompt = self._build_explainability_prompt(shap_data, lime_data)
                summaries["explainability_summary"] = self.generate_summary(exp_prompt)
            
            # Generate bias summary
            if bias_data:
                logger.info("Generating bias summary")
                bias_prompt = self._build_bias_prompt(bias_data)
                summaries["bias_summary"] = self.generate_summary(bias_prompt)
            
            # Generate combined recommendations
            if (shap_data or lime_data) and bias_data:
                logger.info("Generating combined recommendations")
                combined_prompt = self._build_combined_prompt(shap_data, bias_data)
                summaries["recommendations"] = self.generate_summary(combined_prompt)
            
            return summaries
            
        except Exception as e:
            logger.error(f"Combined summary generation failed: {e}")
            return {"error": str(e)}
    
    def _build_explainability_prompt(
        self,
        shap_data: Optional[Dict[str, Any]],
        lime_data: Optional[Dict[str, Any]]
    ) -> str:
        """Build prompt for explainability summary"""
        
        prompt = """You are an ML explainability expert. Your task is to explain model predictions in simple, non-technical language.

**IMPORTANT RULES:**
- Base your analysis ONLY on the data provided below
- Do NOT make up or hallucinate any information
- Keep explanations clear and concise (3-4 sentences)
- Avoid technical jargon
- Focus on practical insights

"""
        
        # Add SHAP data
        if shap_data and "global_feature_importance" in shap_data:
            features = shap_data["global_feature_importance"].get("feature_names", [])
            scores = shap_data["global_feature_importance"].get("importance_scores", [])
            
            prompt += "**Top Features Influencing Predictions (by importance):**\n"
            for feat, score in zip(features[:5], scores[:5]):
                prompt += f"- {feat}: {score:.4f}\n"
            prompt += "\n"
        
        # Add LIME data
        if lime_data and "explanations" in lime_data:
            prompt += "**LIME Feature Contributions (for sample predictions):**\n"
            first_exp = lime_data["explanations"][0] if lime_data["explanations"] else None
            if first_exp and "feature_weights" in first_exp:
                for fw in first_exp["feature_weights"][:5]:
                    prompt += f"- {fw['feature']}: {fw['weight']:.4f}\n"
            prompt += "\n"
        
        prompt += """
**Your Task:**
Provide a brief explanation (3-4 sentences) that:
1. Identifies the most important features driving predictions
2. Explains what this means in practical terms
3. Highlights any notable patterns or insights

Write in plain English for non-technical stakeholders.
"""
        
        return prompt
    
    def _build_bias_prompt(self, bias_data: Dict[str, Any]) -> str:
        """Build prompt for bias analysis summary"""
        
        prompt = """You are an AI fairness analyst. Your task is to analyze model fairness and provide actionable recommendations.

**CRITICAL RULES:**
- Base your analysis ONLY on the metrics provided below
- Do NOT make up or invent any claims
- Provide concrete, actionable recommendations
- Be clear about the severity of any issues
- Keep summary concise (3-4 sentences)

"""
        
        # Add sensitive attribute info
        sensitive_attr = bias_data.get("sensitive_attribute", "unknown")
        groups = bias_data.get("groups", [])
        prompt += f"**Sensitive Attribute:** {sensitive_attr}\n"
        prompt += f"**Groups Analyzed:** {', '.join(map(str, groups))}\n\n"
        
        # Add overall metrics
        overall = bias_data.get("overall_metrics", {})
        disparate_impact = overall.get("disparate_impact", 1.0)
        demographic_parity = overall.get("demographic_parity_difference", 0.0)
        equal_opp = overall.get("equal_opportunity_difference", 0.0)
        
        prompt += "**Fairness Metrics:**\n"
        prompt += f"- Disparate Impact: {disparate_impact:.3f}"
        if disparate_impact < 0.8:
            prompt += " ⚠️ (VIOLATION: Below 0.8 threshold)"
        prompt += "\n"
        prompt += f"- Demographic Parity Difference: {demographic_parity:.3f}"
        if demographic_parity > 0.1:
            prompt += " ⚠️ (VIOLATION: Above 0.1 threshold)"
        prompt += "\n"
        prompt += f"- Equal Opportunity Difference: {equal_opp:.3f}"
        if equal_opp > 0.1:
            prompt += " ⚠️ (VIOLATION: Above 0.1 threshold)"
        prompt += "\n\n"
        
        # Add group metrics
        group_metrics = bias_data.get("group_metrics", {})
        prompt += "**Group-Level Metrics:**\n"
        for group, metrics in group_metrics.items():
            prompt += f"\n{group} (n={metrics.get('group_size', 0)}):\n"
            prompt += f"  - Positive Rate: {metrics.get('positive_rate', 0):.1%}\n"
            prompt += f"  - Accuracy: {metrics.get('accuracy', 0):.1%}\n"
            prompt += f"  - True Positive Rate: {metrics.get('true_positive_rate', 0):.1%}\n"
        
        prompt += """

**Your Task:**
1. **Executive Summary (3-4 sentences):**
   - Overall fairness assessment
   - Which groups are most affected
   - Severity level (low/medium/high concern)

2. **Top 3 Actionable Recommendations (prioritized by impact):**
   Focus on practical steps like:
   - Data collection strategies
   - Reweighting or resampling techniques
   - Feature engineering improvements
   - Threshold adjustments
   - Model retraining approaches

**Format your response as:**

EXECUTIVE SUMMARY:
[Your 3-4 sentence summary]

RECOMMENDATIONS:
1. [Most important action with specific steps]
2. [Second priority with specific steps]
3. [Third priority with specific steps]
"""
        
        return prompt
    
    def _build_combined_prompt(
        self,
        shap_data: Optional[Dict[str, Any]],
        bias_data: Dict[str, Any]
    ) -> str:
        """Build prompt for combined analysis"""
        
        prompt = """You are a senior ML engineer and fairness expert. Provide strategic recommendations based on both model explainability and bias analysis.

**IMPORTANT:**
- Base recommendations ONLY on provided data
- Be specific and actionable
- Prioritize by impact and feasibility

"""
        
        # Add top features from SHAP
        if shap_data and "global_feature_importance" in shap_data:
            features = shap_data["global_feature_importance"].get("feature_names", [])
            prompt += f"**Most Influential Features:** {', '.join(features[:3])}\n"
        
        # Add bias summary
        disparate_impact = bias_data.get("overall_metrics", {}).get("disparate_impact", 1.0)
        sensitive_attr = bias_data.get("sensitive_attribute", "unknown")
        prompt += f"**Disparate Impact ({sensitive_attr}):** {disparate_impact:.3f}\n"
        
        prompt += """

**Task:** Provide 3 prioritized recommendations that address both explainability insights and fairness concerns.

Format:
1. [Action] - [Why] - [How]
2. [Action] - [Why] - [How]
3. [Action] - [Why] - [How]
"""
        
        return prompt


# Global instance
gemini_service = GeminiLLMService()