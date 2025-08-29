"""
Utility functions for the Hill of Towie Wind Turbine Power Prediction project.
"""
import os
import pickle
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from config import MODELS_DIR, SUBMISSIONS_DIR, EXPERIMENT_CONFIG

def setup_logging(log_level: str = "INFO", log_to_file: bool = True, 
                 log_file: str = "experiment.log") -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger("wind_turbine_prediction")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)
    
    return logger

def load_data(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Load data from various file formats."""
    file_path = Path(file_path)
    logger = logging.getLogger("wind_turbine_prediction")
    
    try:
        if file_path.suffix.lower() == '.parquet':
            df = pd.read_parquet(file_path, **kwargs)
        elif file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path, **kwargs)
        elif file_path.suffix.lower() in ['.pkl', '.pickle']:
            with open(file_path, 'rb') as f:
                df = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        logger.info(f"Loaded data from {file_path} - Shape: {df.shape}")
        return df
    
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

def save_data(df: pd.DataFrame, file_path: Union[str, Path], 
              format: str = "parquet", **kwargs) -> None:
    """Save data to various file formats."""
    file_path = Path(file_path)
    logger = logging.getLogger("wind_turbine_prediction")
    
    # Create directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if format.lower() == 'parquet':
            df.to_parquet(file_path, **kwargs)
        elif format.lower() == 'csv':
            df.to_csv(file_path, index=False, **kwargs)
        elif format.lower() in ['pkl', 'pickle']:
            with open(file_path, 'wb') as f:
                pickle.dump(df, f)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved data to {file_path} - Shape: {df.shape}")
    
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {e}")
        raise

def save_model(model: Any, model_name: str, metadata: Optional[Dict] = None) -> Path:
    """Save model with metadata."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{model_name}_{timestamp}.pkl"
    model_path = MODELS_DIR / model_filename
    
    # Create models directory if it doesn't exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        'model': model,
        'metadata': metadata or {},
        'timestamp': timestamp,
        'model_name': model_name
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    logger = logging.getLogger("wind_turbine_prediction")
    logger.info(f"Model saved to {model_path}")
    return model_path

def load_model(model_path: Union[str, Path]) -> Tuple[Any, Dict]:
    """Load model with metadata."""
    model_path = Path(model_path)
    logger = logging.getLogger("wind_turbine_prediction")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    logger.info(f"Model loaded from {model_path}")
    return model_data['model'], model_data.get('metadata', {})

def create_time_features(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    """Create time-based features from datetime column."""
    df = df.copy()
    
    if datetime_col in df.columns:
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
            df[datetime_col] = pd.to_datetime(df[datetime_col])
        
        # Extract time features
        df['year'] = df[datetime_col].dt.year
        df['month'] = df[datetime_col].dt.month
        df['day'] = df[datetime_col].dt.day
        df['hour'] = df[datetime_col].dt.hour
        df['day_of_week'] = df[datetime_col].dt.dayofweek
        df['day_of_year'] = df[datetime_col].dt.dayofyear
        df['week_of_year'] = df[datetime_col].dt.isocalendar().week
        df['quarter'] = df[datetime_col].dt.quarter
        
        # Cyclical features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Is weekend
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    return df

def create_lag_features(df: pd.DataFrame, target_col: str, 
                       lag_periods: List[int]) -> pd.DataFrame:
    """Create lag features for time series data."""
    df = df.copy()
    
    for lag in lag_periods:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    return df

def create_rolling_features(df: pd.DataFrame, columns: List[str], 
                           windows: List[int]) -> pd.DataFrame:
    """Create rolling statistical features."""
    df = df.copy()
    
    for col in columns:
        if col in df.columns:
            for window in windows:
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
                df[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window).min()
                df[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window).max()
    
    return df

def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce memory usage of dataframe by optimizing dtypes."""
    start_mem = df.memory_usage().sum() / 1024**2
    logger = logging.getLogger("wind_turbine_prediction")
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    
    end_mem = df.memory_usage().sum() / 1024**2
    logger.info(f'Memory usage decreased from {start_mem:.2f} MB to {end_mem:.2f} MB '
                f'({100 * (start_mem - end_mem) / start_mem:.2f}% reduction)')
    
    return df

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate common regression metrics."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }
    
    return metrics

def create_submission_file(predictions: np.ndarray, test_ids: np.ndarray, 
                          filename: str = None) -> Path:
    """Create submission file in the required format."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"submission_{timestamp}.csv"
    
    submission_path = SUBMISSIONS_DIR / filename
    
    # Create submissions directory if it doesn't exist
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    submission_df = pd.DataFrame({
        'id': test_ids,
        'power_output': predictions  # This might need to be updated based on actual format
    })
    
    submission_df.to_csv(submission_path, index=False)
    
    logger = logging.getLogger("wind_turbine_prediction")
    logger.info(f"Submission file created: {submission_path}")
    
    return submission_path

def get_feature_importance(model: Any, feature_names: List[str]) -> pd.DataFrame:
    """Extract feature importance from trained model."""
    try:
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_).flatten()
        else:
            raise ValueError("Model does not have feature importance attributes")
        
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df
    
    except Exception as e:
        logger = logging.getLogger("wind_turbine_prediction")
        logger.warning(f"Could not extract feature importance: {e}")
        return pd.DataFrame()

# Initialize logger on module import
logger = setup_logging(
    log_level=EXPERIMENT_CONFIG.log_level,
    log_to_file=EXPERIMENT_CONFIG.log_to_file,
    log_file=EXPERIMENT_CONFIG.log_file
)