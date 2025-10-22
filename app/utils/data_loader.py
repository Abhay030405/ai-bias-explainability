# Load sample data or user-uploaded data

"""
Data loading and validation utilities
"""
import pandas as pd
import numpy as np
from typing import Union, List, Dict
from app.utils.logger import get_logger
from app.utils.constants import FEATURE_NAMES

logger = get_logger()

def load_data_from_dict(data: Union[Dict, List[Dict]]) -> pd.DataFrame:
    """
    Convert JSON data to pandas DataFrame
    
    Args:
        data: Single dict or list of dicts with feature values
        
    Returns:
        DataFrame with validated features
    """
    try:
        # Convert to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            raise ValueError("Data must be dict or list of dicts")
        
        logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")
        return df
    
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise ValueError(f"Data loading failed: {e}")


def validate_features(df: pd.DataFrame, required_features: List[str] = None) -> pd.DataFrame:
    """
    Validate that DataFrame contains required features
    
    Args:
        df: Input DataFrame
        required_features: List of required feature names (default: FEATURE_NAMES)
        
    Returns:
        DataFrame with only required features in correct order
    """
    if required_features is None:
        required_features = FEATURE_NAMES
    
    missing_features = set(required_features) - set(df.columns)
    
    if missing_features:
        logger.error(f"Missing features: {missing_features}")
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Return only required features in correct order
    validated_df = df[required_features].copy()
    
    logger.success(f"Features validated: {len(validated_df)} samples")
    return validated_df


def load_csv_data(file_path: str) -> pd.DataFrame:
    """
    Load data from CSV file
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        DataFrame with loaded data
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        return df
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        raise ValueError(f"CSV loading failed: {e}")


def check_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all features are numeric
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with validated numeric types
    """
    try:
        # Convert all columns to numeric, coerce errors to NaN
        df_numeric = df.apply(pd.to_numeric, errors='coerce')
        
        # Check for NaN values (failed conversions)
        nan_count = df_numeric.isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} non-numeric values, converted to NaN")
        
        # Fill NaN with column mean (simple imputation)
        df_numeric = df_numeric.fillna(df_numeric.mean())
        
        return df_numeric
    
    except Exception as e:
        logger.error(f"Data type validation failed: {e}")
        raise ValueError(f"Invalid data types: {e}")