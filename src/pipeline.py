"""
End-to-end pipeline for Hill of Towie Wind Turbine Power Prediction
"""
import argparse
import sys
from pathlib import Path
import subprocess

from config import MODELS_DIR, SUBMISSIONS_DIR
from utils import setup_logging

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run end-to-end ML pipeline')
    
    # Pipeline control
    parser.add_argument('--mode', type=str, default='full',
                       choices=['train', 'predict', 'submit', 'full'],
                       help='Pipeline mode: train only, predict only, submit only, or full pipeline')
    
    # Training arguments
    parser.add_argument('--model', type=str, default='xgboost',
                       choices=['xgboost', 'lightgbm', 'catboost', 'random_forest', 
                               'linear', 'ridge', 'lasso'],
                       help='Model type to train')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds')
    
    # Prediction arguments
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to trained model (for predict/submit modes)')
    
    # Submission arguments
    parser.add_argument('--submit', action='store_true', default=False,
                       help='Submit predictions to Kaggle after generating them')
    parser.add_argument('--submission-message', type=str, default=None,
                       help='Custom submission message')
    parser.add_argument('--dry-run', action='store_true', default=False,
                       help='Validate submission without actually submitting')
    
    # General
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state for reproducibility')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Verbose output')
    
    return parser.parse_args()

def run_training(model_type, cv_folds, random_state, verbose=True):
    """Run the training pipeline."""
    logger = setup_logging()
    logger.info(f"Starting training with {model_type} model...")
    
    # Build command
    cmd = [
        sys.executable, 'src/train.py',
        '--model', model_type,
        '--cv-folds', str(cv_folds),
        '--random-state', str(random_state),
        '--save-model'
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=not verbose, text=True)
        
        if not verbose and result.stdout:
            logger.info("Training output:")
            logger.info(result.stdout)
        
        logger.info("Training completed successfully!")
        
        # Find the most recent model
        model_files = list(MODELS_DIR.glob(f"{model_type}_model_*.pkl"))
        if model_files:
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Latest model saved: {latest_model}")
            return str(latest_model)
        else:
            logger.warning("No model file found after training")
            return None
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed: {e}")
        if not verbose and e.stderr:
            logger.error(f"Error output: {e.stderr}")
        return None

def run_prediction(model_path, verbose=True):
    """Run the prediction pipeline."""
    logger = setup_logging()
    logger.info(f"Generating predictions using model: {model_path}")
    
    # Build command
    cmd = [
        sys.executable, 'src/predict.py',
        '--model-path', model_path,
        '--save-submission'
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=not verbose, text=True)
        
        if not verbose and result.stdout:
            logger.info("Prediction output:")
            logger.info(result.stdout)
        
        logger.info("Predictions generated successfully!")
        
        # Find the most recent submission file
        submission_files = list(SUBMISSIONS_DIR.glob("predictions_*.csv"))
        if submission_files:
            latest_submission = max(submission_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Latest submission file: {latest_submission}")
            return str(latest_submission)
        else:
            logger.warning("No submission file found after prediction")
            return None
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Prediction failed: {e}")
        if not verbose and e.stderr:
            logger.error(f"Error output: {e.stderr}")
        return None

def run_submission(submission_file, message=None, dry_run=False, verbose=True):
    """Run the submission pipeline."""
    logger = setup_logging()
    logger.info(f"Submitting file: {submission_file}")
    
    # Build command
    cmd = [
        sys.executable, 'src/submit.py',
        '--file', submission_file
    ]
    
    if message:
        cmd.extend(['--message', message])
    
    if dry_run:
        cmd.append('--dry-run')
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=not verbose, text=True)
        
        if not verbose and result.stdout:
            logger.info("Submission output:")
            logger.info(result.stdout)
        
        if dry_run:
            logger.info("Dry run completed successfully!")
        else:
            logger.info("Submission completed successfully!")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Submission failed: {e}")
        if not verbose and e.stderr:
            logger.error(f"Error output: {e.stderr}")
        return False

def find_latest_model(model_type=None):
    """Find the most recent model file."""
    if model_type:
        pattern = f"{model_type}_model_*.pkl"
    else:
        pattern = "*_model_*.pkl"
    
    model_files = list(MODELS_DIR.glob(pattern))
    if model_files:
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        return str(latest_model)
    return None

def main():
    """Main pipeline orchestrator."""
    args = parse_arguments()
    logger = setup_logging()
    
    logger.info(f"Starting {args.mode} pipeline...")
    logger.info(f"Configuration: Model={args.model}, CV folds={args.cv_folds}, Random state={args.random_state}")
    
    # Full pipeline
    if args.mode == 'full':
        # Step 1: Training
        logger.info("=== TRAINING PHASE ===")
        model_path = run_training(args.model, args.cv_folds, args.random_state, args.verbose)
        if not model_path:
            logger.error("Training failed. Aborting pipeline.")
            return False
        
        # Step 2: Prediction
        logger.info("=== PREDICTION PHASE ===")
        submission_file = run_prediction(model_path, args.verbose)
        if not submission_file:
            logger.error("Prediction failed. Aborting pipeline.")
            return False
        
        # Step 3: Submission (if requested)
        if args.submit:
            logger.info("=== SUBMISSION PHASE ===")
            success = run_submission(submission_file, args.submission_message, 
                                   args.dry_run, args.verbose)
            if not success:
                logger.error("Submission failed.")
                return False
        else:
            logger.info("Skipping submission (use --submit flag to submit automatically)")
    
    # Training only
    elif args.mode == 'train':
        logger.info("=== TRAINING PHASE ===")
        model_path = run_training(args.model, args.cv_folds, args.random_state, args.verbose)
        if not model_path:
            logger.error("Training failed.")
            return False
    
    # Prediction only
    elif args.mode == 'predict':
        logger.info("=== PREDICTION PHASE ===")
        
        # Use provided model path or find latest
        model_path = args.model_path
        if not model_path:
            model_path = find_latest_model(args.model)
            if not model_path:
                logger.error("No model found. Please train a model first or specify --model-path")
                return False
        
        submission_file = run_prediction(model_path, args.verbose)
        if not submission_file:
            logger.error("Prediction failed.")
            return False
        
        # Submit if requested
        if args.submit:
            logger.info("=== SUBMISSION PHASE ===")
            success = run_submission(submission_file, args.submission_message,
                                   args.dry_run, args.verbose)
            if not success:
                logger.error("Submission failed.")
                return False
    
    # Submission only
    elif args.mode == 'submit':
        logger.info("=== SUBMISSION PHASE ===")
        
        # Find latest submission file or use provided path
        if args.model_path and Path(args.model_path).suffix == '.csv':
            submission_file = args.model_path
        else:
            submission_files = list(SUBMISSIONS_DIR.glob("predictions_*.csv"))
            if not submission_files:
                logger.error("No submission files found. Generate predictions first.")
                return False
            submission_file = str(max(submission_files, key=lambda x: x.stat().st_mtime))
        
        success = run_submission(submission_file, args.submission_message,
                               args.dry_run, args.verbose)
        if not success:
            logger.error("Submission failed.")
            return False
    
    logger.info("Pipeline completed successfully! 🎉")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)