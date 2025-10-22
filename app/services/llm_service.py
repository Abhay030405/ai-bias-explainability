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
                return "LLM service not available. Please configure GOOGLE_API_KEY."
            
            logger.info("Generating summary with Gemini")
            
            # Generate response
            response = self.model.generate_content(prompt)
            
            # Extract text
            generated_text = response.text
            
            logger.success(f"Generated summary ({len(generated_text)} chars)")
            return generated_text
            
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise RuntimeError(f"LLM generation failed: {e}")
    
    def generate_bias_summary(
        self,
        bias_metrics: Dict[str, Any],
        shap_features: Optional[List[str]] = None
    ) -> str:
        """
        Generate bias analysis summary
        
        Args:
            bias_metrics: Dictionary with fairness metrics
            shap_features: Optional list of top SHAP features
            
        Returns:
            Natural language summary
        """
        try:
            # Build structured prompt
            prompt = self._build_bias_prompt(bias_metrics, shap_features)
            
            # Generate summary
            summary = self.generate_summary(prompt)
            
            return summary
            
        except Exception as e:
            logger.error(f"Bias summary generation failed: {e}")
            raise
    
    def _build_bias_prompt(
        self,
        bias_metrics: Dict[str, Any],
        shap_features: Optional[List[str]] = None
    ) -> str:
        """
        Build structured prompt for bias analysis
        
        Args:
            bias_metrics: Fairness metrics
            shap_features: Top influential features
            
        Returns:
            Formatted prompt string
        """
        prompt = """You are an AI fairness analyst. Your task is to analyze the following model fairness metrics and provide actionable recommendations.

**Important Guidelines:**
- Base your analysis ONLY on the metrics provided below
- Do NOT make up or hallucinate any claims
- Provide concrete, actionable recommendations
- Keep the summary concise (3-4 sentences)
- Provide exactly 3 prioritized recommendations

**Fairness Metrics:**
"""
        
        # Add bias metrics
        if "accuracy_parity" in bias_metrics:
            prompt += f"\nAccuracy Parity: {bias_metrics['accuracy_parity']}"
        
        if "demographic_parity" in bias_metrics:
            prompt += f"\nDemographic Parity: {bias_metrics['demographic_parity']}"
        
        if "equal_opportunity" in bias_metrics:
            prompt += f"\nEqual Opportunity (TPR): {bias_metrics['equal_opportunity']}"
        
        if "disparate_impact" in bias_metrics:
            disparate_impact = bias_metrics['disparate_impact']
            prompt += f"\nDisparate Impact: {disparate_impact:.3f}"
            if disparate_impact < 0.8:
                prompt += " ⚠️ (Below 0.8 threshold - potential bias detected)"
        
        # Add SHAP features if available
        if shap_features:
            prompt += f"\n\n**Top Influential Features (SHAP):**\n"
            for i, feat in enumerate(shap_features[:5], 1):
                prompt += f"{i}. {feat}\n"
        
        prompt += """

**Your Task:**
1. Provide a brief executive summary (3-4 sentences) explaining:
   - Overall fairness assessment
   - Which groups are most affected
   - Severity level (low/medium/high concern)

2. Provide exactly 3 actionable recommendations prioritized by impact:
   - Recommendation 1: [Most important action]
   - Recommendation 2: [Second priority]
   - Recommendation 3: [Third priority]

Focus on practical steps like: data collection, reweighting, resampling, feature engineering, threshold adjustment, or model retraining.

**Format your response as:**

SUMMARY:
[Your 3-4 sentence summary]

RECOMMENDATIONS:
1. [First recommendation]
2. [Second recommendation]
3. [Third recommendation]
"""
        
        return prompt
    
    def generate_explainability_summary(
        self,
        shap_values: Dict[str, Any],
        lime_values: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate explainability summary
        
        Args:
            shap_values: SHAP explanation results
            lime_values: Optional LIME explanation results
            
        Returns:
            Natural language explanation
        """
        try:
            prompt = self._build_explainability_prompt(shap_values, lime_values)
            summary = self.generate_summary(prompt)
            return summary
            
        except Exception as e:
            logger.error(f"Explainability summary generation failed: {e}")
            raise
    
    def _build_explainability_prompt(
        self,
        shap_values: Dict[str, Any],
        lime_values: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build prompt for explainability summary"""
        
        prompt = """You are an ML model explainability expert. Explain the following model predictions in simple terms.

**SHAP Feature Importance:**
"""
        
        # Add SHAP global importance
        if "global_feature_importance" in shap_values:
            features = shap_values["global_feature_importance"].get("feature_names", [])
            scores = shap_values["global_feature_importance"].get("importance_scores", [])
            
            prompt += "\nTop Features Influencing Predictions:\n"
            for feat, score in zip(features[:5], scores[:5]):
                prompt += f"- {feat}: {score:.4f}\n"
        
        prompt += """

**Your Task:**
Provide a clear, non-technical explanation (2-3 sentences) of:
1. Which features have the most impact on predictions
2. What this means in practical terms
3. Any notable patterns or insights

Keep it simple and actionable for non-technical stakeholders.
"""
        
        return prompt


# Global instance
gemini_service = GeminiLLMService()