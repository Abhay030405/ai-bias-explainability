"""
LIME-based explainability service
"""
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from app.utils.logger import get_logger
from app.config import settings

logger = get_logger()

class LIMEExplainer:
    """LIME explainability computation"""
    
    def __init__(self, model, training_data: pd.DataFrame, feature_names: List[str], mode: str = "classification"):
        """
        Initialize LIME explainer
        
        Args:
            model: Trained ML model
            training_data: Training dataset for LIME
            feature_names: List of feature names
            mode: 'classification' or 'regression'
        """
        self.model = model
        self.training_data = training_data
        self.feature_names = feature_names
        self.mode = mode
        self.explainer = None
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize LIME tabular explainer"""
        try:
            logger.info(f"Initializing LIME explainer in {self.mode} mode")
            
            # Create LIME explainer
            self.explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=self.training_data.values,
                feature_names=self.feature_names,
                mode=self.mode,
                discretize_continuous=True,
                random_state=42
            )
            
            logger.success("LIME explainer initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize LIME explainer: {e}")
            raise RuntimeError(f"LIME initialization failed: {e}")
    
    def explain_instance(self, instance: pd.Series, num_features: int = None) -> Dict[str, Any]:
        """
        Explain a single instance using LIME
        
        Args:
            instance: Single row as pandas Series
            num_features: Number of features to include in explanation
            
        Returns:
            Dictionary with LIME explanation
        """
        try:
            if num_features is None:
                num_features = settings.lime_num_features
            
            logger.info(f"Computing LIME explanation for instance with {num_features} features")
            
            # Get prediction function
            if hasattr(self.model, 'predict_proba'):
                predict_fn = self.model.predict_proba
            else:
                predict_fn = self.model.predict
            
            # Explain instance
            explanation = self.explainer.explain_instance(
                data_row=instance.values,
                predict_fn=predict_fn,
                num_features=num_features,
                num_samples=settings.lime_num_samples
            )
            
            # Extract explanation as list of (feature, weight) tuples
            explanation_list = explanation.as_list()
            
            # Parse explanations
            feature_weights = []
            for feat_desc, weight in explanation_list:
                feature_weights.append({
                    "feature": feat_desc,
                    "weight": float(weight)
                })
            
            # Get prediction probability
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(instance.values.reshape(1, -1))
                prediction_proba = float(proba[0][1]) if len(proba[0]) > 1 else float(proba[0][0])
            else:
                prediction_proba = None
            
            result = {
                "feature_weights": feature_weights,
                "prediction_probability": prediction_proba,
                "num_features_explained": len(explanation_list)
            }
            
            logger.success("LIME explanation completed")
            return result
            
        except Exception as e:
            logger.error(f"LIME explanation failed: {e}")
            raise RuntimeError(f"LIME explanation failed: {e}")
    
    def explain_batch(self, data: pd.DataFrame, num_features: int = None) -> List[Dict[str, Any]]:
        """
        Explain multiple instances
        
        Args:
            data: DataFrame with multiple rows
            num_features: Number of features per explanation
            
        Returns:
            List of LIME explanations
        """
        try:
            explanations = []
            
            for idx, row in data.iterrows():
                explanation = self.explain_instance(row, num_features)
                explanations.append(explanation)
            
            logger.info(f"Completed LIME explanations for {len(explanations)} instances")
            return explanations
            
        except Exception as e:
            logger.error(f"Batch LIME explanation failed: {e}")
            raise RuntimeError(f"Batch LIME failed: {e}")


def compute_lime_explanation(
    model: Any,
    input_data: pd.DataFrame,
    training_data: pd.DataFrame,
    feature_names: List[str],
    mode: str = "classification"
) -> Dict[str, Any]:
    """
    Main function to compute LIME explanations
    
    Args:
        model: Trained model
        input_data: Data to explain
        training_data: Training data for LIME
        feature_names: Feature names
        mode: 'classification' or 'regression'
        
    Returns:
        Dictionary with LIME explanations
    """
    try:
        explainer = LIMEExplainer(model, training_data, feature_names, mode)
        
        # Explain all instances
        if len(input_data) == 1:
            explanations = [explainer.explain_instance(input_data.iloc[0])]
        else:
            explanations = explainer.explain_batch(input_data)
        
        return {
            "explanations": explanations,
            "num_instances": len(input_data),
            "explainer_type": "LIME"
        }
        
    except Exception as e:
        logger.error(f"LIME explanation pipeline failed: {e}")
        raise

    