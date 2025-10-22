"""
SHAP-based explainability service
"""
import shap
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from app.utils.logger import get_logger
from app.config import settings

logger = get_logger()

class SHAPExplainer:
    """SHAP explainability computation"""
    
    def __init__(self, model, background_data: pd.DataFrame = None):
        """
        Initialize SHAP explainer
        
        Args:
            model: Trained ML model
            background_data: Background dataset for SHAP (optional)
        """
        self.model = model
        self.background_data = background_data
        self.explainer = None
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize appropriate SHAP explainer based on model type"""
        try:
            model_type = type(self.model).__name__
            logger.info(f"Initializing SHAP explainer for model type: {model_type}")
            
            # Check if model is in a pipeline
            if hasattr(self.model, 'named_steps'):
                # Extract the actual model from pipeline
                actual_model = self.model.named_steps.get('classifier') or \
                             self.model.named_steps.get('regressor') or \
                             list(self.model.named_steps.values())[-1]
            else:
                actual_model = self.model
            
            # Try TreeExplainer first (fast for tree-based models)
            try:
                self.explainer = shap.TreeExplainer(actual_model)
                logger.success("Using TreeExplainer (fast)")
            except Exception:
                # Fallback to general Explainer
                logger.info("TreeExplainer not available, using general Explainer")
                if self.background_data is not None:
                    # Use a sample of background data
                    sample_size = min(len(self.background_data), settings.shap_sample_size)
                    background_sample = shap.sample(self.background_data, sample_size)
                    self.explainer = shap.Explainer(self.model.predict, background_sample)
                else:
                    self.explainer = shap.Explainer(self.model.predict)
                logger.success("Using general Explainer")
                
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            raise RuntimeError(f"SHAP initialization failed: {e}")
    
    def explain_instance(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute SHAP values for input samples
        
        Args:
            X: Input DataFrame with features
            
        Returns:
            Dictionary with SHAP values and metadata
        """
        try:
            logger.info(f"Computing SHAP values for {len(X)} samples")
            
            # Compute SHAP values
            shap_values = self.explainer(X)
            
            # Extract values based on SHAP version
            if hasattr(shap_values, 'values'):
                values = shap_values.values
                base_values = shap_values.base_values
            else:
                values = shap_values
                base_values = self.explainer.expected_value
            
            # Handle multi-class case (take positive class)
            if len(values.shape) > 2:
                values = values[:, :, 1]
                if isinstance(base_values, np.ndarray) and len(base_values.shape) > 1:
                    base_values = base_values[:, 1]
            
            # Prepare response
            result = {
                "shap_values": values.tolist(),
                "base_value": float(base_values[0]) if isinstance(base_values, np.ndarray) else float(base_values),
                "feature_names": list(X.columns),
                "num_samples": len(X)
            }
            
            logger.success("SHAP computation completed")
            return result
            
        except Exception as e:
            logger.error(f"SHAP computation failed: {e}")
            raise RuntimeError(f"SHAP explanation failed: {e}")
    
    def get_feature_importance(self, X: pd.DataFrame, top_k: int = 10) -> Dict[str, List]:
        """
        Get global feature importance from SHAP values
        
        Args:
            X: Input DataFrame
            top_k: Number of top features to return
            
        Returns:
            Dictionary with feature importance rankings
        """
        try:
            shap_values = self.explainer(X)
            
            if hasattr(shap_values, 'values'):
                values = shap_values.values
            else:
                values = shap_values
            
            # Handle multi-class
            if len(values.shape) > 2:
                values = values[:, :, 1]
            
            # Compute mean absolute SHAP values
            mean_abs_shap = np.abs(values).mean(axis=0)
            
            # Get feature names
            feature_names = list(X.columns)
            
            # Sort by importance
            importance_indices = np.argsort(mean_abs_shap)[::-1][:top_k]
            
            result = {
                "feature_names": [feature_names[i] for i in importance_indices],
                "importance_scores": [float(mean_abs_shap[i]) for i in importance_indices]
            }
            
            logger.info(f"Computed global feature importance (top {top_k})")
            return result
            
        except Exception as e:
            logger.error(f"Feature importance computation failed: {e}")
            raise RuntimeError(f"Feature importance failed: {e}")


def compute_shap_explanation(
    model: Any,
    input_data: pd.DataFrame,
    background_data: pd.DataFrame = None
) -> Dict[str, Any]:
    """
    Main function to compute SHAP explanations
    
    Args:
        model: Trained model
        input_data: Data to explain
        background_data: Background dataset (optional)
        
    Returns:
        Dictionary with SHAP explanations
    """
    try:
        explainer = SHAPExplainer(model, background_data)
        
        # Local explanations
        local_explanations = explainer.explain_instance(input_data)
        
        # Global feature importance
        feature_importance = explainer.get_feature_importance(input_data)
        
        return {
            "local_explanations": local_explanations,
            "global_feature_importance": feature_importance,
            "explainer_type": "SHAP"
        }
        
    except Exception as e:
        logger.error(f"SHAP explanation pipeline failed: {e}")
        raise