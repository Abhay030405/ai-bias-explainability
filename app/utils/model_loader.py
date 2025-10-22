"""
Model loading utilities
"""
import joblib
import json
from pathlib import Path
from typing import Any, Dict
from app.utils.logger import get_logger
from app.utils.constants import MODEL_PATH, MODEL_INFO_PATH

logger = get_logger()

class ModelLoader:
    """Singleton class to load and cache ML model"""
    
    _instance = None
    _model = None
    _model_info = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance
    
    def load_model(self, model_path: str = MODEL_PATH) -> Any:
        """
        Load trained model from pickle file
        
        Args:
            model_path: Path to model pickle file
            
        Returns:
            Loaded model object
        """
        if self._model is None:
            try:
                logger.info(f"Loading model from {model_path}")
                self._model = joblib.load(model_path)
                logger.success("Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise RuntimeError(f"Model loading failed: {e}")
        
        return self._model
    
    def load_model_info(self, info_path: str = MODEL_INFO_PATH) -> Dict:
        """
        Load model metadata
        
        Args:
            info_path: Path to model info JSON file
            
        Returns:
            Dictionary with model metadata
        """
        if self._model_info is None:
            try:
                if Path(info_path).exists():
                    with open(info_path, 'r') as f:
                        self._model_info = json.load(f)
                    logger.info("Model info loaded")
                else:
                    logger.warning(f"Model info not found at {info_path}")
                    self._model_info = {}
            except Exception as e:
                logger.error(f"Failed to load model info: {e}")
                self._model_info = {}
        
        return self._model_info
    
    def get_feature_names(self) -> list:
        """Get feature names from model info"""
        info = self.load_model_info()
        return info.get('feature_names', [])
    
    def reload_model(self, model_path: str = MODEL_PATH):
        """Force reload model (useful for model updates)"""
        self._model = None
        self._model_info = None
        return self.load_model(model_path)


# Global instance
model_loader = ModelLoader()