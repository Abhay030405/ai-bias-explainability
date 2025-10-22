 # Thresholds, feature names, etc.
"""
Constants and configuration values for FairLens AI
"""

# Feature names (must match training data)
FEATURE_NAMES = [
    'age',
    'income', 
    'loan_amount',
    'credit_score',
    'employment_years',
    'debt_to_income'
]

# Sensitive attributes for fairness analysis
SENSITIVE_ATTRIBUTES = ['gender', 'ethnicity', 'age_group', 'race']

# Model paths
MODEL_PATH = "models/model.pkl"
MODEL_INFO_PATH = "models/model_info.json"

# Fairness thresholds
DISPARATE_IMPACT_THRESHOLD = 0.8  # 80% rule
DEMOGRAPHIC_PARITY_THRESHOLD = 0.1  # 10% difference threshold
EQUAL_OPPORTUNITY_THRESHOLD = 0.1  # 10% difference in TPR
MIN_GROUP_SIZE = 30  # Minimum samples per group for reliable metrics

# SHAP configuration
SHAP_SAMPLE_SIZE = 100  # Number of samples for background data
SHAP_MAX_DISPLAY = 10  # Max features to show in plots

# LIME configuration
LIME_NUM_FEATURES = 10  # Number of features for LIME explanation
LIME_NUM_SAMPLES = 5000  # Samples for LIME perturbation

# API configuration
API_TITLE = "FairLens AI - Explainability & Bias Detection API"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
FairLens AI provides comprehensive ML model explainability and fairness analysis.

## Features:
* **Explainability**: SHAP and LIME-based feature importance
* **Bias Detection**: Demographic parity, equal opportunity, disparate impact
* **LLM Summaries**: Natural language explanations and recommendations
"""

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"

# Report storage
REPORTS_DIR = "reports"
LOGS_DIR = "logs"