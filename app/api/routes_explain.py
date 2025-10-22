"""
API routes for model explainability (SHAP & LIME)
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd

from app.services.shap_service import compute_shap_explanation
from app.services.lime_service import compute_lime_explanation
from app.utils.data_loader import load_data_from_dict, validate_features
from app.utils.logger import get_logger
from app.dependencies import get_model, get_feature_names
from app.config import settings

logger = get_logger()
router = APIRouter(prefix="/api", tags=["Explainability"])


# Pydantic models for request/response
class ExplainRequest(BaseModel):
    """Request model for explainability endpoint"""
    data: List[Dict[str, Any]] = Field(..., description="Input data samples to explain")
    explainer_type: Optional[str] = Field("both", description="Type of explainer: 'shap', 'lime', or 'both'")
    
    class Config:
        json_schema_extra = {
            "example": {
                "data": [
                    {
                        "age": 35,
                        "income": 75000,
                        "loan_amount": 25000,
                        "credit_score": 720,
                        "employment_years": 8,
                        "debt_to_income": 0.25
                    }
                ],
                "explainer_type": "both"
            }
        }


class ExplainResponse(BaseModel):
    """Response model for explainability endpoint"""
    status: str
    num_samples: int
    shap_explanation: Optional[Dict[str, Any]] = None
    lime_explanation: Optional[Dict[str, Any]] = None
    message: str


@router.post("/explain", response_model=ExplainResponse)
async def explain_predictions(
    request: ExplainRequest,
    model = Depends(get_model),
    feature_names: List[str] = Depends(get_feature_names)
):
    """
    Explain model predictions using SHAP and/or LIME
    
    - **data**: List of input samples with feature values
    - **explainer_type**: Choose 'shap', 'lime', or 'both' (default)
    
    Returns explanations showing which features influenced the prediction and by how much.
    """
    try:
        logger.info(f"Received explain request with {len(request.data)} samples")
        
        # Convert input to DataFrame
        input_df = load_data_from_dict(request.data)
        
        # Validate features
        validated_df = validate_features(input_df, feature_names)
        
        # Load background/training data for explainers
        try:
            background_df = pd.read_csv(f"{settings.data_dir}/train.csv")
            background_df = validate_features(background_df, feature_names)
            # Sample if too large
            if len(background_df) > settings.shap_sample_size:
                background_df = background_df.sample(n=settings.shap_sample_size, random_state=42)
        except Exception as e:
            logger.warning(f"Could not load background data: {e}. Using input data as background.")
            background_df = validated_df
        
        # Initialize response
        shap_result = None
        lime_result = None
        
        # Compute SHAP explanation
        if request.explainer_type in ["shap", "both"]:
            try:
                logger.info("Computing SHAP explanation")
                shap_result = compute_shap_explanation(
                    model=model,
                    input_data=validated_df,
                    background_data=background_df
                )
                logger.success("SHAP explanation completed")
            except Exception as e:
                logger.error(f"SHAP computation failed: {e}")
                if request.explainer_type == "shap":
                    raise HTTPException(status_code=500, detail=f"SHAP computation failed: {str(e)}")
        
        # Compute LIME explanation
        if request.explainer_type in ["lime", "both"]:
            try:
                logger.info("Computing LIME explanation")
                lime_result = compute_lime_explanation(
                    model=model,
                    input_data=validated_df,
                    training_data=background_df,
                    feature_names=feature_names,
                    mode="classification"
                )
                logger.success("LIME explanation completed")
            except Exception as e:
                logger.error(f"LIME computation failed: {e}")
                if request.explainer_type == "lime":
                    raise HTTPException(status_code=500, detail=f"LIME computation failed: {str(e)}")
        
        # Prepare response
        response = ExplainResponse(
            status="success",
            num_samples=len(validated_df),
            shap_explanation=shap_result,
            lime_explanation=lime_result,
            message=f"Successfully explained {len(validated_df)} predictions using {request.explainer_type}"
        )
        
        logger.success(f"Explain request completed successfully")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Explain endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@router.get("/explain/features")
async def get_model_features(feature_names: List[str] = Depends(get_feature_names)):
    """
    Get list of features expected by the model
    
    Returns the feature names and their order that the model expects.
    """
    return {
        "feature_names": feature_names,
        "num_features": len(feature_names),
        "message": "These are the features required for prediction"
    }