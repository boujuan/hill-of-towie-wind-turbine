"""
Configuration settings for the Hill of Towie Wind Turbine Power Prediction project.
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_DATA_DIR = DATA_DIR / "train"
TEST_DATA_DIR = DATA_DIR / "test"
MODELS_DIR = PROJECT_ROOT / "models"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
CONFIGS_DIR = PROJECT_ROOT / "configs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Data file paths
TRAIN_FILE = TRAIN_DATA_DIR / "training_dataset.parquet"
TEST_FILE = TEST_DATA_DIR / "submission_dataset.parquet"
SAMPLE_SUBMISSION_FILE = RAW_DATA_DIR / "sample_model_submission.csv"

# Kaggle competition settings
COMPETITION_NAME = "hill-of-towie-wind-turbine-power-prediction"

@dataclass
class ModelConfig:
    """Configuration for model training and inference."""
    
    # Model selection
    model_type: str = "xgboost"  # Options: xgboost, lightgbm, catboost, sklearn, neural_network
    
    # Cross-validation settings
    n_folds: int = 5
    random_state: int = 42
    
    # Feature engineering
    create_lag_features: bool = True
    lag_periods: List[int] = field(default_factory=lambda: [1, 2, 3, 6, 12, 24])
    create_rolling_features: bool = True
    rolling_windows: List[int] = field(default_factory=lambda: [3, 6, 12, 24])
    create_time_features: bool = True
    
    # Model hyperparameters (defaults for XGBoost)
    model_params: Dict[str, Any] = field(default_factory=lambda: {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 1000,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1
    })
    
    # Training settings
    early_stopping_rounds: int = 100
    verbose: bool = True

@dataclass
class DataConfig:
    """Configuration for data processing."""
    
    # Target column
    target_column: str = "power_output"  # This might need to be updated based on actual data
    
    # Date/time columns
    datetime_column: Optional[str] = "timestamp"  # This might need to be updated
    
    # Columns to exclude from training
    exclude_columns: List[str] = field(default_factory=lambda: [
        "id", "timestamp"  # Common columns to exclude
    ])
    
    # Data preprocessing
    handle_missing_values: bool = True
    missing_value_strategy: str = "median"  # Options: mean, median, mode, drop
    remove_outliers: bool = True
    outlier_method: str = "iqr"  # Options: iqr, zscore, isolation_forest
    
    # Feature scaling
    scale_features: bool = True
    scaling_method: str = "standard"  # Options: standard, minmax, robust

@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking."""
    
    experiment_name: str = "wind_turbine_power_prediction"
    track_experiments: bool = True
    save_predictions: bool = True
    save_feature_importance: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file: str = "experiment.log"

@dataclass
class SubmissionConfig:
    """Configuration for Kaggle submissions."""
    
    submission_message: str = "Automated submission"
    create_ensemble: bool = False
    ensemble_weights: Optional[List[float]] = None

# Default configuration
DEFAULT_CONFIG = ModelConfig()
DATA_CONFIG = DataConfig()
EXPERIMENT_CONFIG = ExperimentConfig()
SUBMISSION_CONFIG = SubmissionConfig()

def get_config(config_type: str = "default") -> Any:
    """Get configuration object by type."""
    configs = {
        "model": DEFAULT_CONFIG,
        "data": DATA_CONFIG,
        "experiment": EXPERIMENT_CONFIG,
        "submission": SUBMISSION_CONFIG
    }
    return configs.get(config_type, DEFAULT_CONFIG)

def update_config_from_dict(config_obj: Any, updates: Dict[str, Any]) -> Any:
    """Update configuration object with dictionary values."""
    for key, value in updates.items():
        if hasattr(config_obj, key):
            setattr(config_obj, key, value)
    return config_obj

# Environment-specific settings
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, TRAIN_DATA_DIR, 
                 TEST_DATA_DIR, MODELS_DIR, SUBMISSIONS_DIR, CONFIGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)