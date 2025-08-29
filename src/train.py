"""
Training script for Hill of Towie Wind Turbine Power Prediction
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import joblib
import warnings
warnings.filterwarnings('ignore')

# ML Models
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from config import (
    TRAIN_FILE, TEST_FILE, MODELS_DIR, PROCESSED_DATA_DIR,
    ModelConfig, DataConfig, EXPERIMENT_CONFIG
)
from utils import (
    load_data, save_model, setup_logging, calculate_metrics,
    create_time_features, create_lag_features, create_rolling_features,
    reduce_memory_usage, get_feature_importance
)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train wind turbine power prediction model')
    parser.add_argument('--model', type=str, default='xgboost',
                       choices=['xgboost', 'lightgbm', 'catboost', 'random_forest', 
                               'linear', 'ridge', 'lasso'],
                       help='Model type to train')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state for reproducibility')
    parser.add_argument('--save-model', action='store_true', default=True,
                       help='Save trained model')
    parser.add_argument('--feature-engineering', action='store_true', default=True,
                       help='Apply feature engineering')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (YAML)')
    
    return parser.parse_args()

def identify_target_column(df):
    """Identify the target column in the dataset."""
    # Look for columns with 'power' in the name
    power_cols = [col for col in df.columns if 'power' in col.lower()]
    if power_cols:
        return power_cols[0]
    
    # Look for columns with 'target' in the name
    target_cols = [col for col in df.columns if 'target' in col.lower()]
    if target_cols:
        return target_cols[0]
    
    # If no obvious target, assume last numerical column
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        return numerical_cols[-1]
    
    raise ValueError("Could not identify target column")

def identify_datetime_column(df):
    """Identify datetime column in the dataset."""
    datetime_candidates = []
    
    for col in df.columns:
        if any(word in col.lower() for word in ['time', 'date', 'timestamp']):
            datetime_candidates.append(col)
    
    # Try to find the actual datetime column
    for col in datetime_candidates:
        try:
            pd.to_datetime(df[col].iloc[:100])  # Test on first 100 rows
            return col
        except:
            continue
    
    return None

def preprocess_data(df, config, target_col=None, is_train=True, scaler=None, imputer=None):
    """Preprocess the data according to configuration."""
    logger = setup_logging()
    df = df.copy()
    
    # Identify datetime column
    datetime_col = identify_datetime_column(df)
    
    # Create time features if datetime column exists
    if datetime_col and config.create_time_features:
        logger.info(f"Creating time features from {datetime_col}")
        df = create_time_features(df, datetime_col)
    
    # Create lag and rolling features for training data
    if is_train and target_col and target_col in df.columns:
        if config.create_lag_features and datetime_col:
            logger.info("Creating lag features")
            df = create_lag_features(df, target_col, config.lag_periods)
        
        if config.create_rolling_features and datetime_col:
            logger.info("Creating rolling features")
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in numerical_cols:
                numerical_cols.remove(target_col)
            df = create_rolling_features(df, numerical_cols[:5], config.rolling_windows)
    
    # Separate features and target
    if target_col and is_train:
        # Remove target and any excluded columns
        exclude_cols = config.exclude_columns + ([datetime_col] if datetime_col else [])
        feature_cols = [col for col in df.columns if col not in exclude_cols + [target_col]]
        X = df[feature_cols].copy()
        y = df[target_col].copy()
    else:
        exclude_cols = config.exclude_columns + ([datetime_col] if datetime_col else [])
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols].copy()
        y = None
    
    # Handle missing values
    if config.handle_missing_values:
        if is_train:
            if config.missing_value_strategy == 'median':
                imputer = SimpleImputer(strategy='median')
            elif config.missing_value_strategy == 'mean':
                imputer = SimpleImputer(strategy='mean')
            else:
                imputer = SimpleImputer(strategy='most_frequent')
            
            X_imputed = pd.DataFrame(
                imputer.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            if imputer is not None:
                X_imputed = pd.DataFrame(
                    imputer.transform(X),
                    columns=X.columns,
                    index=X.index
                )
            else:
                X_imputed = X.copy()
        X = X_imputed
    
    # Feature scaling
    if config.scale_features:
        if is_train:
            if config.scaling_method == 'standard':
                scaler = StandardScaler()
            elif config.scaling_method == 'robust':
                scaler = RobustScaler()
            else:
                scaler = MinMaxScaler()
            
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            if scaler is not None:
                X_scaled = pd.DataFrame(
                    scaler.transform(X),
                    columns=X.columns,
                    index=X.index
                )
            else:
                X_scaled = X.copy()
        X = X_scaled
    
    # Reduce memory usage
    X = reduce_memory_usage(X)
    
    if is_train:
        return X, y, scaler, imputer
    else:
        return X

def get_model(model_type, model_params, random_state=42):
    """Get model instance based on type and parameters."""
    model_params = model_params.copy()
    model_params['random_state'] = random_state
    
    if model_type == 'xgboost':
        model = xgb.XGBRegressor(**model_params)
    elif model_type == 'lightgbm':
        # Convert XGBoost params to LightGBM params
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': model_params.get('learning_rate', 0.1),
            'feature_fraction': model_params.get('colsample_bytree', 0.8),
            'bagging_fraction': model_params.get('subsample', 0.8),
            'n_estimators': model_params.get('n_estimators', 1000),
            'random_state': random_state,
            'verbosity': -1
        }
        model = lgb.LGBMRegressor(**lgb_params)
    elif model_type == 'catboost':
        cb_params = {
            'loss_function': 'RMSE',
            'iterations': model_params.get('n_estimators', 1000),
            'learning_rate': model_params.get('learning_rate', 0.1),
            'depth': model_params.get('max_depth', 6),
            'random_state': random_state,
            'verbose': False
        }
        model = cb.CatBoostRegressor(**cb_params)
    elif model_type == 'random_forest':
        rf_params = {
            'n_estimators': model_params.get('n_estimators', 100),
            'max_depth': model_params.get('max_depth', None),
            'random_state': random_state,
            'n_jobs': -1
        }
        model = RandomForestRegressor(**rf_params)
    elif model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'ridge':
        model = Ridge(alpha=1.0, random_state=random_state)
    elif model_type == 'lasso':
        model = Lasso(alpha=1.0, random_state=random_state)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model

def train_model(X, y, model_type, model_config, cv_folds=5, random_state=42):
    """Train model with cross-validation."""
    logger = setup_logging()
    
    # Get model
    model = get_model(model_type, model_config.model_params, random_state)
    
    # Cross-validation
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Calculate cross-validation scores
    cv_scores = cross_val_score(model, X, y, cv=cv, 
                               scoring='neg_root_mean_squared_error', n_jobs=-1)
    
    logger.info(f"Cross-validation RMSE: {-cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Train final model on full dataset
    logger.info("Training final model on full dataset...")
    model.fit(X, y)
    
    # Calculate training metrics
    y_pred = model.predict(X)
    train_metrics = calculate_metrics(y, y_pred)
    
    logger.info("Training metrics:")
    for metric, value in train_metrics.items():
        logger.info(f"  {metric.upper()}: {value:.4f}")
    
    return model, train_metrics, -cv_scores.mean()

def main():
    """Main training pipeline."""
    args = parse_arguments()
    logger = setup_logging()
    
    # Load configuration
    model_config = ModelConfig()
    data_config = DataConfig()
    
    # Update config with command line arguments
    model_config.model_type = args.model
    model_config.n_folds = args.cv_folds
    model_config.random_state = args.random_state
    
    logger.info(f"Starting training pipeline with {args.model} model")
    logger.info(f"Configuration: CV folds={args.cv_folds}, Random state={args.random_state}")
    
    # Load data
    logger.info("Loading training data...")
    train_df = load_data(TRAIN_FILE)
    
    # Identify target column
    target_col = identify_target_column(train_df)
    logger.info(f"Target column identified: {target_col}")
    
    # Update data config
    data_config.target_column = target_col
    datetime_col = identify_datetime_column(train_df)
    if datetime_col:
        data_config.datetime_column = datetime_col
        logger.info(f"Datetime column identified: {datetime_col}")
    
    # Preprocess data
    logger.info("Preprocessing data...")
    X, y, scaler, imputer = preprocess_data(train_df, data_config, target_col, is_train=True)
    
    logger.info(f"Final feature matrix shape: {X.shape}")
    logger.info(f"Target variable shape: {y.shape}")
    
    # Train model
    logger.info(f"Training {args.model} model...")
    model, train_metrics, cv_score = train_model(
        X, y, args.model, model_config, args.cv_folds, args.random_state
    )
    
    # Get feature importance
    feature_importance = get_feature_importance(model, X.columns.tolist())
    if not feature_importance.empty:
        logger.info("Top 10 most important features:")
        for idx, row in feature_importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Save model if requested
    if args.save_model:
        metadata = {
            'model_type': args.model,
            'cv_score': cv_score,
            'train_metrics': train_metrics,
            'feature_names': X.columns.tolist(),
            'target_column': target_col,
            'datetime_column': datetime_col,
            'scaler': scaler,
            'imputer': imputer,
            'data_config': data_config,
            'model_config': model_config
        }
        
        model_path = save_model(model, f"{args.model}_model", metadata)
        logger.info(f"Model saved to: {model_path}")
        
        # Save feature importance
        if not feature_importance.empty:
            importance_path = MODELS_DIR / f"feature_importance_{args.model}.csv"
            feature_importance.to_csv(importance_path, index=False)
            logger.info(f"Feature importance saved to: {importance_path}")
    
    logger.info("Training pipeline completed successfully!")
    
    return model, train_metrics, cv_score

if __name__ == "__main__":
    main()