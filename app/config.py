# Configurations (model paths, API URLs)

"""
Configuration management for FairLens AI
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    api_title: str = "FairLens AI - Explainability & Bias Detection"
    api_version: str = "1.0.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    google_api_key: str | None = None
    llm_max_tokens: int | None = 1024
    
    # Model paths
    model_path: str = "models/model.pkl"
    model_info_path: str = "models/model_info.json"
    
    # Directory paths
    data_dir: str = "data"
    reports_dir: str = "reports"
    logs_dir: str = "logs"
    
    # LLM Configuration (for Phase 3)
    openai_api_key: Optional[str] = None
    llm_temperature: float = 0.1
    llm_model: str = "gpt-3.5-turbo"
    
    # SHAP Configuration
    shap_sample_size: int = 100
    shap_max_display: int = 10
    
    # LIME Configuration
    lime_num_features: int = 10
    lime_num_samples: int = 5000
    
    # Fairness thresholds
    disparate_impact_threshold: float = 0.8
    demographic_parity_threshold: float = 0.1
    equal_opportunity_threshold: float = 0.1
    min_group_size: int = 30
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()

# Create necessary directories
Path(settings.reports_dir).mkdir(exist_ok=True)
Path(settings.logs_dir).mkdir(exist_ok=True)