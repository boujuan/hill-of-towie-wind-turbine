#!/usr/bin/env python3
"""
Setup script for Hill of Towie Wind Turbine Power Prediction project.
Handles environment setup and dependency installation.
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description=None):
    """Run a command and handle errors."""
    if description:
        print(f"\n🔧 {description}...")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True)
        if result.stdout.strip():
            print(result.stdout.strip())
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def install_basic_requirements():
    """Install basic requirements without heavy ML libraries."""
    print("📦 Installing basic requirements...")
    
    basic_packages = [
        "pandas>=2.0.0",
        "numpy>=1.24.0", 
        "pyarrow>=12.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0",
        "jupyter>=1.0.0",
        "ipykernel>=6.25.0",
        "tqdm>=4.65.0",
        "joblib>=1.3.0",
        "kaggle>=1.5.16",
        "optuna>=3.3.0",
        "shap>=0.42.0"
    ]
    
    for package in basic_packages:
        success = run_command(f"pip install '{package}'", f"Installing {package.split('>=')[0]}")
        if not success:
            print(f"⚠️  Warning: Failed to install {package}")

def install_pytorch_rocm():
    """Install PyTorch with ROCm support for AMD GPUs."""
    print("\n🎯 Installing PyTorch with ROCm support...")
    
    # Set temporary directory to home to avoid /tmp space issues
    home_tmp = Path.home() / "tmp"
    home_tmp.mkdir(exist_ok=True)
    
    env = os.environ.copy()
    env['TMPDIR'] = str(home_tmp)
    env['TEMP'] = str(home_tmp)
    env['TMP'] = str(home_tmp)
    
    cmd = "pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2"
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True, env=env)
        print("✅ PyTorch with ROCm installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install PyTorch with ROCm: {e}")
        print("You can try installing CPU version instead:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        return False

def install_tensorflow():
    """Install TensorFlow CPU version."""
    print("\n🧠 Installing TensorFlow (CPU version)...")
    return run_command("pip install tensorflow-cpu", "Installing TensorFlow CPU")

def install_optional_packages():
    """Install optional packages that might fail."""
    optional_packages = [
        ("catboost>=1.2.0", "CatBoost"),
        ("statsmodels>=0.14.0", "Statsmodels"),
        ("prophet>=1.1.0", "Prophet"),
        ("ydata-profiling>=4.0.0", "YData Profiling"),
        ("great-expectations>=0.17.0", "Great Expectations")
    ]
    
    print("\n📋 Installing optional packages...")
    for package, name in optional_packages:
        success = run_command(f"pip install '{package}'", f"Installing {name}")
        if not success:
            print(f"⚠️  Skipping {name} - install manually if needed")

def setup_jupyter_kernel():
    """Setup Jupyter kernel for the environment."""
    print("\n📓 Setting up Jupyter kernel...")
    kernel_name = "wind-turbine-prediction"
    display_name = "Hill of Towie Wind Turbine"
    
    success = run_command(
        f"python -m ipykernel install --user --name {kernel_name} --display-name '{display_name}'",
        "Installing Jupyter kernel"
    )
    
    if success:
        print(f"✅ Jupyter kernel '{display_name}' installed!")
        print(f"   You can now select it in Jupyter notebooks")

def check_environment():
    """Check if the environment is set up correctly."""
    print("\n🔍 Checking environment setup...")
    
    checks = [
        ("python --version", "Python version"),
        ("pip --version", "Pip version"), 
        ("jupyter --version", "Jupyter version"),
        ("kaggle --version", "Kaggle API"),
        ("python -c 'import pandas; print(f\"Pandas {pandas.__version__}\")'", "Pandas"),
        ("python -c 'import sklearn; print(f\"Scikit-learn {sklearn.__version__}\")'", "Scikit-learn"),
        ("python -c 'import xgboost; print(f\"XGBoost {xgboost.__version__}\")'", "XGBoost"),
        ("python -c 'import lightgbm; print(f\"LightGBM {lightgbm.__version__}\")'", "LightGBM"),
    ]
    
    for cmd, name in checks:
        success = run_command(cmd, f"Checking {name}")
        if not success:
            print(f"⚠️  {name} not available")

def main():
    parser = argparse.ArgumentParser(description='Setup Hill of Towie Wind Turbine prediction environment')
    parser.add_argument('--skip-pytorch', action='store_true', 
                       help='Skip PyTorch installation (due to disk space)')
    parser.add_argument('--skip-tensorflow', action='store_true',
                       help='Skip TensorFlow installation')
    parser.add_argument('--skip-optional', action='store_true',
                       help='Skip optional packages')
    parser.add_argument('--basic-only', action='store_true',
                       help='Install only basic requirements')
    
    args = parser.parse_args()
    
    print("🚀 Setting up Hill of Towie Wind Turbine Power Prediction Environment")
    print("=" * 70)
    
    # Install basic requirements
    install_basic_requirements()
    
    if not args.basic_only:
        # Install PyTorch if not skipped
        if not args.skip_pytorch:
            install_pytorch_rocm()
        
        # Install TensorFlow if not skipped  
        if not args.skip_tensorflow:
            install_tensorflow()
        
        # Install optional packages if not skipped
        if not args.skip_optional:
            install_optional_packages()
    
    # Setup Jupyter kernel
    setup_jupyter_kernel()
    
    # Check environment
    check_environment()
    
    print("\n" + "=" * 70)
    print("🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Start Jupyter Lab: jupyter lab")
    print("2. Open notebooks/01_eda.ipynb to begin analysis")
    print("3. Run training: python src/train.py --model xgboost")
    print("4. Generate predictions: python src/predict.py --model-path models/your_model.pkl")
    print("5. Submit to Kaggle: python src/submit.py --file submissions/your_submission.csv")
    
    if args.skip_pytorch or args.skip_tensorflow:
        print("\n⚠️  Note: Some deep learning libraries were skipped.")
        print("   Install them manually if needed:")
        if args.skip_pytorch:
            print("   - PyTorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2")
        if args.skip_tensorflow:
            print("   - TensorFlow: pip install tensorflow-cpu")

if __name__ == "__main__":
    main()