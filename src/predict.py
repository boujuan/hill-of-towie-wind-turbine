"""
Prediction script for Hill of Towie Wind Turbine Power Prediction
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from config import TEST_FILE, MODELS_DIR, SUBMISSIONS_DIR
from utils import load_data, load_model, setup_logging, create_submission_file
from train import preprocess_data, identify_datetime_column

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate predictions for wind turbine power')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--test-data', type=str, default=None,
                       help='Path to test data (default: use TEST_FILE from config)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path for predictions')
    parser.add_argument('--save-submission', action='store_true', default=True,
                       help='Save predictions in submission format')
    
    return parser.parse_args()

def load_test_data(test_data_path=None):
    """Load test data."""
    if test_data_path:
        test_df = load_data(test_data_path)
    else:
        test_df = load_data(TEST_FILE)
    return test_df

def generate_predictions(model, test_df, metadata):
    """Generate predictions for test data."""
    logger = setup_logging()
    
    # Extract preprocessing components from metadata
    scaler = metadata.get('scaler')
    imputer = metadata.get('imputer')
    data_config = metadata.get('data_config')
    feature_names = metadata.get('feature_names', [])
    target_column = metadata.get('target_column')
    datetime_column = metadata.get('datetime_column')
    
    logger.info(f"Preprocessing test data with target column: {target_column}")
    
    # Preprocess test data
    X_test = preprocess_data(
        test_df, 
        data_config, 
        target_col=target_column,
        is_train=False, 
        scaler=scaler, 
        imputer=imputer
    )
    
    logger.info(f"Test feature matrix shape: {X_test.shape}")
    
    # Ensure feature alignment
    if feature_names:
        # Check if all required features are present
        missing_features = set(feature_names) - set(X_test.columns)
        if missing_features:
            logger.warning(f"Missing features in test data: {missing_features}")
            # Add missing features with zeros
            for feature in missing_features:
                X_test[feature] = 0
        
        # Select and order features to match training
        X_test = X_test[feature_names]
        logger.info(f"Aligned test features shape: {X_test.shape}")
    
    # Generate predictions
    logger.info("Generating predictions...")
    predictions = model.predict(X_test)
    
    logger.info(f"Generated {len(predictions)} predictions")
    logger.info(f"Prediction stats - Min: {predictions.min():.4f}, Max: {predictions.max():.4f}, Mean: {predictions.mean():.4f}")
    
    return predictions

def create_predictions_dataframe(test_df, predictions, metadata):
    """Create a dataframe with predictions."""
    
    # Try to identify ID column
    id_columns = [col for col in test_df.columns if 'id' in col.lower()]
    if id_columns:
        id_col = id_columns[0]
        ids = test_df[id_col].values
    else:
        # Use index as ID if no ID column found
        ids = test_df.index.values
    
    target_column = metadata.get('target_column', 'power_output')
    
    predictions_df = pd.DataFrame({
        'id': ids,
        target_column: predictions
    })
    
    return predictions_df

def save_predictions(predictions_df, output_path, logger):
    """Save predictions to file."""
    if output_path is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = SUBMISSIONS_DIR / f"predictions_{timestamp}.csv"
    else:
        output_path = Path(output_path)
    
    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save predictions
    predictions_df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to: {output_path}")
    
    return output_path

def validate_predictions(predictions, logger):
    """Validate predictions for common issues."""
    issues = []
    
    # Check for NaN values
    if np.isnan(predictions).any():
        nan_count = np.isnan(predictions).sum()
        issues.append(f"Found {nan_count} NaN values in predictions")
    
    # Check for infinite values
    if np.isinf(predictions).any():
        inf_count = np.isinf(predictions).sum()
        issues.append(f"Found {inf_count} infinite values in predictions")
    
    # Check for negative values (if power output should be positive)
    if (predictions < 0).any():
        neg_count = (predictions < 0).sum()
        issues.append(f"Found {neg_count} negative values in predictions")
    
    # Check for very large values
    if (predictions > 1e6).any():
        large_count = (predictions > 1e6).sum()
        issues.append(f"Found {large_count} very large values (>1e6) in predictions")
    
    if issues:
        logger.warning("Prediction validation issues:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("Predictions passed validation checks")
    
    return issues

def main():
    """Main prediction pipeline."""
    args = parse_arguments()
    logger = setup_logging()
    
    logger.info("Starting prediction pipeline...")
    logger.info(f"Model path: {args.model_path}")
    
    # Load trained model
    logger.info("Loading trained model...")
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model, metadata = load_model(model_path)
    logger.info(f"Model loaded successfully")
    logger.info(f"Model type: {metadata.get('model_type', 'Unknown')}")
    logger.info(f"CV score: {metadata.get('cv_score', 'N/A')}")
    
    # Load test data
    logger.info("Loading test data...")
    test_df = load_test_data(args.test_data)
    logger.info(f"Test data shape: {test_df.shape}")
    
    # Generate predictions
    predictions = generate_predictions(model, test_df, metadata)
    
    # Validate predictions
    validation_issues = validate_predictions(predictions, logger)
    
    # Create predictions dataframe
    predictions_df = create_predictions_dataframe(test_df, predictions, metadata)
    
    # Save predictions
    if args.save_submission:
        output_path = save_predictions(predictions_df, args.output, logger)
        
        # Display sample predictions
        logger.info("Sample predictions:")
        logger.info(f"\n{predictions_df.head(10).to_string(index=False)}")
        
        return output_path, predictions_df
    else:
        return None, predictions_df

if __name__ == "__main__":
    main()