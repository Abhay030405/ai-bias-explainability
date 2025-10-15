# Load ML model (.pkl)
import pickle
def load_model(model_path):
    """Load a machine learning model from a .pkl file."""
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model
# Load data from a CSV file
import pandas as pd
def load_data(data_path):
    """Load data from a CSV file."""
    return pd.read_csv(data_path)
# Load configuration settings from a JSON file
import json
def load_config(config_path):
    """Load configuration settings from a JSON file."""
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config
# Load a natural language model for explanations
from langchain.llms import OpenAI
def load_natural_language_model(model_name):
    """Load a natural language model for explanations."""
    return OpenAI(model_name=model_name)
# Load sample data or user-uploaded data
def load_sample_data(sample_path):
    """Load sample data or user-uploaded data."""
    return pd.read_csv(sample_path)
# Load a machine learning model from a .pkl file

# Shared utils (model loader, helper functions)
import os
def load_shared_utils():
    """Load shared utilities for the application."""
    utils_path = os.path.join(os.path.dirname(__file__), 'utils')
    return utils_path
# Configurations (model paths, API URLs)
def load_configurations(config_file):
    """Load configurations such as model paths and API URLs."""
    with open(config_file, 'r') as file:
        return json.load(file)
    