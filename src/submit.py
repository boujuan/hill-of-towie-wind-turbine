"""
Kaggle submission script for Hill of Towie Wind Turbine Power Prediction
"""
import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

from config import COMPETITION_NAME, SUBMISSIONS_DIR, SAMPLE_SUBMISSION_FILE
from utils import load_data, setup_logging

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Submit predictions to Kaggle competition')
    parser.add_argument('--file', '-f', type=str, required=True,
                       help='Path to submission file')
    parser.add_argument('--message', '-m', type=str, default=None,
                       help='Submission message')
    parser.add_argument('--validate-format', action='store_true', default=True,
                       help='Validate submission format before submitting')
    parser.add_argument('--dry-run', action='store_true', default=False,
                       help='Validate submission without actually submitting')
    
    return parser.parse_args()

def validate_submission_format(submission_file, sample_submission_file=None):
    """Validate submission file format against sample submission."""
    logger = setup_logging()
    
    # Load submission file
    try:
        submission_df = pd.read_csv(submission_file)
    except Exception as e:
        logger.error(f"Could not read submission file: {e}")
        return False, str(e)
    
    # Load sample submission for format validation
    if sample_submission_file is None:
        sample_submission_file = SAMPLE_SUBMISSION_FILE
    
    try:
        sample_df = load_data(sample_submission_file)
    except Exception as e:
        logger.warning(f"Could not load sample submission for validation: {e}")
        # Continue without sample validation
        sample_df = None
    
    issues = []
    
    # Basic checks
    if submission_df.empty:
        issues.append("Submission file is empty")
    
    # Check for required columns if we have sample submission
    if sample_df is not None:
        sample_columns = set(sample_df.columns)
        submission_columns = set(submission_df.columns)
        
        # Check if columns match
        missing_columns = sample_columns - submission_columns
        extra_columns = submission_columns - sample_columns
        
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
        
        if extra_columns:
            issues.append(f"Unexpected columns: {extra_columns}")
        
        # Check number of rows
        if len(submission_df) != len(sample_df):
            issues.append(f"Row count mismatch. Expected: {len(sample_df)}, Got: {len(submission_df)}")
        
        # Check for ID column alignment if present
        id_cols = [col for col in sample_df.columns if 'id' in col.lower()]
        if id_cols:
            id_col = id_cols[0]
            if id_col in submission_df.columns:
                sample_ids = set(sample_df[id_col].values)
                submission_ids = set(submission_df[id_col].values)
                
                missing_ids = sample_ids - submission_ids
                extra_ids = submission_ids - sample_ids
                
                if missing_ids:
                    issues.append(f"Missing IDs: {len(missing_ids)} IDs not found in submission")
                
                if extra_ids:
                    issues.append(f"Extra IDs: {len(extra_ids)} unexpected IDs in submission")
    
    # Check for missing values in prediction columns
    prediction_cols = [col for col in submission_df.columns if 'id' not in col.lower()]
    for col in prediction_cols:
        null_count = submission_df[col].isnull().sum()
        if null_count > 0:
            issues.append(f"Column '{col}' has {null_count} missing values")
        
        # Check for infinite values
        if pd.api.types.is_numeric_dtype(submission_df[col]):
            inf_count = submission_df[col].isin([float('inf'), float('-inf')]).sum()
            if inf_count > 0:
                issues.append(f"Column '{col}' has {inf_count} infinite values")
    
    # Print validation results
    if issues:
        logger.error("Submission validation failed:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False, "; ".join(issues)
    else:
        logger.info("Submission validation passed!")
        logger.info(f"Submission shape: {submission_df.shape}")
        logger.info(f"Columns: {submission_df.columns.tolist()}")
        
        # Show sample of predictions
        if len(submission_df) > 0:
            logger.info("Sample predictions:")
            logger.info(f"\n{submission_df.head().to_string(index=False)}")
        
        return True, "Validation passed"

def check_kaggle_authentication():
    """Check if Kaggle API is properly authenticated."""
    logger = setup_logging()
    
    try:
        # Try to list competitions to test authentication
        result = subprocess.run(
            ['kaggle', 'competitions', 'list'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            logger.info("Kaggle authentication successful")
            return True
        else:
            logger.error(f"Kaggle authentication failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("Kaggle authentication check timed out")
        return False
    except Exception as e:
        logger.error(f"Error checking Kaggle authentication: {e}")
        return False

def submit_to_kaggle(submission_file, message=None):
    """Submit file to Kaggle competition."""
    logger = setup_logging()
    
    # Generate default message if none provided
    if message is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"Automated submission - {timestamp}"
    
    logger.info(f"Submitting to competition: {COMPETITION_NAME}")
    logger.info(f"Submission file: {submission_file}")
    logger.info(f"Message: {message}")
    
    try:
        # Submit to Kaggle
        cmd = [
            'kaggle', 'competitions', 'submit',
            '-c', COMPETITION_NAME,
            '-f', str(submission_file),
            '-m', message
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            logger.info("Submission successful!")
            logger.info(f"Kaggle output: {result.stdout}")
            return True, result.stdout
        else:
            logger.error(f"Submission failed: {result.stderr}")
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        logger.error("Submission timed out")
        return False, "Submission timed out"
    except Exception as e:
        logger.error(f"Error during submission: {e}")
        return False, str(e)

def get_submission_history():
    """Get submission history for the competition."""
    logger = setup_logging()
    
    try:
        result = subprocess.run(
            ['kaggle', 'competitions', 'submissions', COMPETITION_NAME],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            logger.info("Recent submissions:")
            logger.info(result.stdout)
            return True, result.stdout
        else:
            logger.error(f"Could not retrieve submission history: {result.stderr}")
            return False, result.stderr
            
    except Exception as e:
        logger.error(f"Error retrieving submission history: {e}")
        return False, str(e)

def main():
    """Main submission pipeline."""
    args = parse_arguments()
    logger = setup_logging()
    
    logger.info("Starting Kaggle submission pipeline...")
    
    # Validate submission file exists
    submission_file = Path(args.file)
    if not submission_file.exists():
        logger.error(f"Submission file not found: {submission_file}")
        return False
    
    # Validate format if requested
    if args.validate_format:
        logger.info("Validating submission format...")
        format_valid, validation_message = validate_submission_format(submission_file)
        
        if not format_valid:
            logger.error("Submission format validation failed. Aborting.")
            return False
    
    # Check Kaggle authentication
    if not args.dry_run:
        logger.info("Checking Kaggle authentication...")
        if not check_kaggle_authentication():
            logger.error("Kaggle authentication failed. Please check your credentials.")
            return False
    
    # Dry run - just validate without submitting
    if args.dry_run:
        logger.info("Dry run completed successfully. File is ready for submission.")
        return True
    
    # Submit to Kaggle
    success, message = submit_to_kaggle(submission_file, args.message)
    
    if success:
        logger.info("Submission completed successfully!")
        
        # Get updated submission history
        logger.info("Fetching updated submission history...")
        get_submission_history()
        
        return True
    else:
        logger.error(f"Submission failed: {message}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)