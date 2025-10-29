"""
API routes for LLM-based summaries and explanations
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from app.services.llm_service import gemini_service as summary_service
from app.utils.logger import get_logger

logger = get_logger()
router = APIRouter(prefix="/api", tags=["LLM Summaries"])


# Pydantic models
class SummaryRequest(BaseModel):
    """Request model for summary generation"""
    shap_explanation: Optional[Dict[str, Any]] = Field(None, description="SHAP explanation results")
    lime_explanation: Optional[Dict[str, Any]] = Field(None, description="LIME explanation results")
    bias_results: Optional[Dict[str, Any]] = Field(None, description="Bias detection results")
    
    class Config:
        json_schema_extra = {
            "example": {
                "shap_explanation": {
                    "global_feature_importance": {
                        "feature_names": ["credit_score", "income", "loan_amount"],
                        "importance_scores": [0.45, 0.32, 0.23]
                    }
                },
                "bias_results": {
                    "sensitive_attribute": "gender",
                    "groups": ["Male", "Female"],
                    "overall_metrics": {
                        "disparate_impact": 0.75,
                        "demographic_parity_difference": 0.15
                    },
                    "group_metrics": {
                        "Male": {"group_size": 50, "positive_rate": 0.6, "accuracy": 0.85},
                        "Female": {"group_size": 50, "positive_rate": 0.45, "accuracy": 0.82}
                    }
                }
            }
        }


class SummaryResponse(BaseModel):
    """Response model for summary generation"""
    status: str
    explainability_summary: Optional[str] = None
    bias_summary: Optional[str] = None
    recommendations: Optional[str] = None
    message: str


@router.post("/summary", response_model=SummaryResponse)
async def generate_summary(request: SummaryRequest):
    """
    Generate natural language summaries from explainability and bias results
    
    - **shap_explanation**: SHAP explanation data (optional)
    - **lime_explanation**: LIME explanation data (optional)
    - **bias_results**: Bias detection results (optional)
    
    Returns LLM-generated summaries including:
    - Explainability summary (if SHAP/LIME provided)
    - Bias analysis summary (if bias results provided)
    - Combined recommendations (if both provided)
    
    Uses Google Gemini for natural language generation.
    """
    try:
        logger.info("Received summary generation request")
        
        # Check if LLM is available
        if not gemini_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="LLM service not available. Please configure GOOGLE_API_KEY in environment variables."
            )
        
        # Validate input
        if not any([request.shap_explanation, request.lime_explanation, request.bias_results]):
            raise HTTPException(
                status_code=400,
                detail="At least one of shap_explanation, lime_explanation, or bias_results must be provided"
            )
        
        # Generate summaries
        logger.info("Generating LLM summaries...")
        summaries = summary_service.generate_combined_summary(
            shap_data=request.shap_explanation,
            lime_data=request.lime_explanation,
            bias_data=request.bias_results
        )
                
        # Check for errors
        if "error" in summaries:
            raise HTTPException(status_code=500, detail=f"Summary generation failed: {summaries['error']}")
        
        # Prepare response
        response = SummaryResponse(
            status="success",
            explainability_summary=summaries.get("explainability_summary"),
            bias_summary=summaries.get("bias_summary"),
            recommendations=summaries.get("recommendations"),
            message="Summary generated successfully using Gemini LLM"
        )
        
        logger.success("Summary generation completed")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")


@router.get("/summary/status")
async def get_llm_status():
    """
    Check LLM service availability and configuration
    
    Returns information about the LLM service status.
    """
    is_available = gemini_service.is_available()
    
    return {
        "llm_available": is_available,
        "llm_model": "gemini-pro" if is_available else None,
        "message": "LLM service is ready" if is_available else "LLM service not configured. Set GOOGLE_API_KEY in .env file",
        "api_key_configured": is_available
    }


@router.post("/summary/bias-only")
async def generate_bias_summary(bias_results: Dict[str, Any]):
    """
    Generate summary for bias results only
    
    Quick endpoint for bias analysis summaries.
    """
    try:
        if not gemini_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="LLM service not available"
            )
        
        logger.info("Generating bias-only summary")
        prompt = summary_service._build_bias_prompt(bias_results)
        summary = summary_service.generate_summary(prompt)
        return {
            "status": "success",
            "summary": summary,
            "message": "Bias summary generated"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bias summary failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/summary/explainability-only")
async def generate_explainability_summary(
    shap_explanation: Optional[Dict[str, Any]] = None,
    lime_explanation: Optional[Dict[str, Any]] = None
):
    """
    Generate summary for explainability results only
    
    Quick endpoint for SHAP/LIME summaries.
    """
    try:
        if not gemini_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="LLM service not available"
            )
        
        if not shap_explanation and not lime_explanation:
            raise HTTPException(
                status_code=400,
                detail="Provide at least shap_explanation or lime_explanation"
            )
        
        logger.info("Generating explainability-only summary")
        prompt = summary_service._build_explainability_prompt(shap_explanation, lime_explanation)
        summary = summary_service.generate_summary(prompt)
        return {
            "status": "success",
            "summary": summary,
            "message": "Explainability summary generated"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Explainability summary failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))