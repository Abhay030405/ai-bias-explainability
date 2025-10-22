"""
Preprocessing utilities for input data
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from app.utils.logger import get_logger

logger = get_logger()


def preprocess_input(data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply preprocessing steps to input data
    (This should match the preprocessing done during training)
    
    Args:
        data: Raw input DataFrame
        
    Returns:
        Preprocessed DataFrame
    """
    try:
        df = data.copy()
        
        # Handle missing values
        if df.isna().any().any():
            logger.warning("Missing values detected, filling with column means")
            df = df.fillna(df.mean())
        
        # Clip outliers (simple approach)
        for col in df.select_dtypes(include=[np.number]).columns:
            q1 = df[col].quantile(0.01)
            q99 = df[col].quantile(0.99)
            df[col] = df[col].clip(q1, q99)
        
        logger.info("Preprocessing completed")
        return df
    
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise ValueError(f"Preprocessing error: {e}")


def extract_features_and_sensitive_attr(
    data: pd.DataFrame, 
    feature_names: list,
    sensitive_attr: Optional[str] = None
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Separate features and sensitive attributes
    
    Args:
        data: Input DataFrame
        feature_names: List of feature column names
        sensitive_attr: Name of sensitive attribute column
        
    Returns:
        Tuple of (features_df, sensitive_series)
    """
    try:
        # Extract features
        X = data[feature_names].copy()
        
        # Extract sensitive attribute if provided
        sensitive = None
        if sensitive_attr and sensitive_attr in data.columns:
            sensitive = data[sensitive_attr].copy()
            logger.info(f"Extracted sensitive attribute: {sensitive_attr}")
        
        return X, sensitive
    
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        raise ValueError(f"Feature extraction error: {e}")


def prepare_for_prediction(input_data: Dict[str, Any], feature_names: list) -> pd.DataFrame:
    """
    Prepare input data for model prediction
    
    Args:
        input_data: Dictionary with feature values
        feature_names: Expected feature names
        
    Returns:
        DataFrame ready for prediction
    """
    try:
        # Create DataFrame
        df = pd.DataFrame([input_data])
        
        # Ensure all features present
        for feat in feature_names:
            if feat not in df.columns:
                raise ValueError(f"Missing feature: {feat}")
        
        # Select and order features
        df = df[feature_names]
        
        # Ensure numeric types
        df = df.apply(pd.to_numeric, errors='coerce')
        
        # Fill any NaN
        df = df.fillna(0)
        
        logger.info(f"Prepared {len(df)} samples for prediction")
        return df
    
    except Exception as e:
        logger.error(f"Prediction preparation failed: {e}")
        raise ValueError(f"Preparation error: {e}")