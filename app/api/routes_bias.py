"""
API routes for bias and fairness detection
"""
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import io

from app.services.bias_service import compute_bias_metrics
from app.utils.data_loader import load_data_from_dict
from app.utils.logger import get_logger
from app.dependencies import get_model, get_feature_names
from app.config import settings

logger = get_logger()
router = APIRouter(prefix="/api", tags=["Bias Detection"])


# Pydantic models for request/response
class BiasRequest(BaseModel):
    """Request model for bias detection endpoint"""
    data: List[Dict[str, Any]] = Field(..., description="Dataset with features, true labels, and sensitive attribute")
    sensitive_attr: str = Field(..., description="Name of sensitive attribute column (e.g., 'gender', 'ethnicity')")
    true_label_col: str = Field(default="true_label", description="Name of true label column")
    
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
                        "debt_to_income": 0.25,
                        "gender": "Male",
                        "true_label": 1
                    },
                    {
                        "age": 28,
                        "income": 45000,
                        "loan_amount": 15000,
                        "credit_score": 650,
                        "employment_years": 3,
                        "debt_to_income": 0.45,
                        "gender": "Female",
                        "true_label": 0
                    }
                ],
                "sensitive_attr": "gender",
                "true_label_col": "true_label"
            }
        }


class BiasResponse(BaseModel):
    """Response model for bias detection endpoint"""
    status: str
    sensitive_attribute: str
    num_samples: int
    groups: List[str]
    group_metrics: Dict[str, Any]
    overall_metrics: Dict[str, Any]
    warnings: List[str]
    message: str


@router.post("/bias", response_model=BiasResponse)
async def detect_bias(
    request: BiasRequest,
    model = Depends(get_model),
    feature_names: List[str] = Depends(get_feature_names)
):
    """
    Detect bias and compute fairness metrics for model predictions
    
    - **data**: List of samples with features, true labels, and sensitive attribute
    - **sensitive_attr**: Name of sensitive attribute (e.g., 'gender', 'ethnicity')
    - **true_label_col**: Name of true label column (default: 'true_label')
    
    Returns fairness metrics including:
    - Demographic parity (positive rate per group)
    - Equal opportunity (True Positive Rate per group)
    - Accuracy parity (accuracy by group)
    - Disparate impact (ratio of positive rates)
    - False Positive/Negative rates per group
    """
    try:
        logger.info(f"Received bias detection request with {len(request.data)} samples")
        logger.info(f"Sensitive attribute: {request.sensitive_attr}")
        
        # Convert input to DataFrame
        input_df = load_data_from_dict(request.data)
        
        # Validate required columns
        if request.sensitive_attr not in input_df.columns:
            raise ValueError(f"Sensitive attribute '{request.sensitive_attr}' not found in data")
        
        if request.true_label_col not in input_df.columns:
            raise ValueError(f"True label column '{request.true_label_col}' not found in data")
        
        # Check if all features are present
        missing_features = set(feature_names) - set(input_df.columns)
        if missing_features:
            logger.warning(f"Missing features: {missing_features}. Will use available features only.")
            # Use only available features
            available_features = [f for f in feature_names if f in input_df.columns]
            if not available_features:
                raise ValueError("No valid features found in data")
            feature_names = available_features
        
        # Compute bias metrics
        logger.info("Computing bias metrics...")
        results = compute_bias_metrics(
            model=model,
            data=input_df,
            feature_names=feature_names,
            true_label_col=request.true_label_col,
            sensitive_attr=request.sensitive_attr
        )
        
        # Prepare response
        response = BiasResponse(
            status="success",
            sensitive_attribute=results["sensitive_attribute"],
            num_samples=len(input_df),
            groups=results["groups"],
            group_metrics=results["group_metrics"],
            overall_metrics=results["overall_metrics"],
            warnings=results["warnings"],
            message=f"Bias analysis completed for {len(results['groups'])} groups"
        )
        
        logger.success("Bias detection completed successfully")
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Bias detection endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=f"Bias detection failed: {str(e)}")


@router.get("/bias/thresholds")
async def get_fairness_thresholds():
    """
    Get configured fairness thresholds
    
    Returns the thresholds used for detecting bias violations.
    """
    return {
        "disparate_impact_threshold": settings.disparate_impact_threshold,
        "demographic_parity_threshold": settings.demographic_parity_threshold,
        "equal_opportunity_threshold": settings.equal_opportunity_threshold,
        "min_group_size": settings.min_group_size,
        "description": {
            "disparate_impact": "Ratio of min to max positive rates (80% rule: should be >= 0.8)",
            "demographic_parity": "Maximum difference in positive rates across groups",
            "equal_opportunity": "Maximum difference in TPR across groups",
            "min_group_size": "Minimum samples per group for reliable metrics"
        }
    }


@router.post("/bias/upload", response_model=BiasResponse)
async def detect_bias_from_csv(
    file: UploadFile = File(..., description="CSV file with features, true labels, and sensitive attribute"),
    sensitive_attr: str = Form(..., description="Name of sensitive attribute column"),
    true_label_col: str = Form(default="true_label", description="Name of true label column"),
    model = Depends(get_model),
    feature_names: List[str] = Depends(get_feature_names)
):
    """
    Detect bias from uploaded CSV file
    
    - **file**: CSV file containing dataset
    - **sensitive_attr**: Name of sensitive attribute column (e.g., 'gender', 'ethnicity')
    - **true_label_col**: Name of true label column (default: 'true_label')
    
    Upload a CSV file with your dataset. The file must contain:
    - All feature columns
    - True label column
    - Sensitive attribute column
    
    Example CSV structure:
    ```
    age,income,loan_amount,credit_score,employment_years,debt_to_income,gender,true_label
    35,75000,25000,720,8,0.25,Male,1
    28,45000,15000,650,3,0.45,Female,0
    ```
    """
    try:
        logger.info(f"Received CSV file upload: {file.filename}")
        logger.info(f"Sensitive attribute: {sensitive_attr}")
        
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV file")
        
        # Read CSV file
        contents = await file.read()
        csv_data = io.StringIO(contents.decode('utf-8'))
        
        try:
            df = pd.read_csv(csv_data)
            logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")
        
        # Validate required columns
        if sensitive_attr not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Sensitive attribute '{sensitive_attr}' not found in CSV. Available columns: {list(df.columns)}"
            )
        
        if true_label_col not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"True label column '{true_label_col}' not found in CSV. Available columns: {list(df.columns)}"
            )
        
        # Check for required features
        missing_features = set(feature_names) - set(df.columns)
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            available_features = [f for f in feature_names if f in df.columns]
            if not available_features:
                raise HTTPException(
                    status_code=400,
                    detail=f"No valid features found. Required: {feature_names}"
                )
            feature_names = available_features
        
        # Compute bias metrics
        logger.info("Computing bias metrics from CSV...")
        results = compute_bias_metrics(
            model=model,
            data=df,
            feature_names=feature_names,
            true_label_col=true_label_col,
            sensitive_attr=sensitive_attr
        )
        
        # Prepare response
        response = BiasResponse(
            status="success",
            sensitive_attribute=results["sensitive_attribute"],
            num_samples=len(df),
            groups=results["groups"],
            group_metrics=results["group_metrics"],
            overall_metrics=results["overall_metrics"],
            warnings=results["warnings"],
            message=f"Bias analysis completed for {len(results['groups'])} groups from CSV file"
        )
        
        logger.success(f"CSV bias detection completed: {file.filename}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CSV bias detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Bias detection from CSV failed: {str(e)}")