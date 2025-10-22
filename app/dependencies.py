# Shared utils (model loader, helper functions)

"""
Shared dependencies for FastAPI routes
"""
from typing import Any
from app.utils.model_loader import model_loader
from app.utils.logger import get_logger
from app.config import settings

logger = get_logger()

def get_model() -> Any:
    """
    Dependency to get loaded model
    
    Returns:
        Loaded ML model
    """
    try:
        model = model_loader.load_model(settings.model_path)
        return model
    except Exception as e:
        logger.error(f"Failed to load model in dependency: {e}")
        raise RuntimeError("Model not available")


def get_feature_names() -> list:
    """
    Dependency to get feature names
    
    Returns:
        List of feature names
    """
    try:
        feature_names = model_loader.get_feature_names()
        if not feature_names:
            logger.warning("Feature names not found in model info, using defaults")
            from app.utils.constants import FEATURE_NAMES
            return FEATURE_NAMES
        return feature_names
    except Exception as e:
        logger.error(f"Failed to get feature names: {e}")
        from app.utils.constants import FEATURE_NAMES
        return FEATURE_NAMES