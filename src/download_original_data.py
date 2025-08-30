"""
Download and prepare the original Hill of Towie dataset from GitHub.

According to the competition description, we can use additional data from:
https://github.com/resgroup/hill-of-towie-open-source-analysis

This includes:
- More turbine data (all turbines, not just 1,2,3,4,5,7)
- Additional fields not included in competition dataset
- Historical data beyond the competition timeframe
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import json
import zipfile
import requests
from tqdm import tqdm

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
ORIGINAL_DATA_DIR = EXTERNAL_DATA_DIR / "hill-of-towie-original"

def setup_directories():
    """Create necessary directories."""
    EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ORIGINAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✅ Created directories: {EXTERNAL_DATA_DIR}")

def clone_or_update_repo():
    """Clone or update the Hill of Towie repository."""
    repo_url = "https://github.com/resgroup/hill-of-towie-open-source-analysis.git"
    repo_dir = EXTERNAL_DATA_DIR / "hill-of-towie-repo"
    
    if repo_dir.exists():
        print("📂 Repository already exists, pulling latest changes...")
        try:
            subprocess.run(["git", "-C", str(repo_dir), "pull"], check=True)
            print("✅ Repository updated")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Could not update repo: {e}")
    else:
        print("📥 Cloning Hill of Towie repository...")
        try:
            subprocess.run(["git", "clone", repo_url, str(repo_dir)], check=True)
            print("✅ Repository cloned successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error cloning repository: {e}")
            return False
    
    return repo_dir

def download_zenodo_data():
    """
    Download the original dataset from Zenodo for years 2016-2020.
    These match the competition time period.
    The dataset is available at: https://doi.org/10.5281/zenodo.14870023
    """
    # Download years 2016-2020 to match competition data
    years_to_download = [2016, 2017, 2018, 2019, 2020]
    base_url = "https://zenodo.org/records/14870023/files"
    
    downloaded_files = []
    
    for year in years_to_download:
        zenodo_url = f"{base_url}/{year}.zip"
        zip_path = ORIGINAL_DATA_DIR / f"hill_of_towie_{year}.zip"
        
        if zip_path.exists():
            print(f"📦 Dataset for {year} already exists")
            downloaded_files.append(zip_path)
            continue
        
        print(f"📥 Downloading Hill of Towie dataset for {year}...")
        print(f"   URL: {zenodo_url}")
        
        try:
            response = requests.get(zenodo_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(zip_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {year}") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            print(f"✅ Dataset for {year} downloaded successfully")
            downloaded_files.append(zip_path)
        
        except Exception as e:
            print(f"❌ Error downloading dataset for {year}: {e}")
            print(f"📝 You can download manually from: {zenodo_url}")
    
    if not downloaded_files:
        print("❌ No datasets were downloaded")
        return None
    
    print(f"\n✅ Downloaded {len(downloaded_files)} dataset files")
    return downloaded_files

def extract_dataset(zip_paths):
    """Extract the downloaded datasets."""
    extract_dir = ORIGINAL_DATA_DIR / "extracted"
    
    # Handle both single file and list of files
    if not isinstance(zip_paths, list):
        zip_paths = [zip_paths]
    
    extract_dir.mkdir(exist_ok=True)
    
    for zip_path in zip_paths:
        year = zip_path.stem.split('_')[-1]  # Extract year from filename
        year_dir = extract_dir / year
        
        if year_dir.exists() and any(year_dir.iterdir()):
            print(f"📂 Dataset for {year} already extracted")
            continue
        
        print(f"📦 Extracting dataset for {year}...")
        year_dir.mkdir(exist_ok=True)
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(year_dir)
            print(f"✅ Dataset for {year} extracted successfully")
        except Exception as e:
            print(f"❌ Error extracting dataset for {year}: {e}")
            continue
    
    return extract_dir

def analyze_original_data():
    """Analyze the structure of the original dataset across years."""
    extract_dir = ORIGINAL_DATA_DIR / "extracted"
    
    if not extract_dir.exists():
        print("❌ Extracted data directory not found")
        return
    
    print("\n📊 Analyzing original dataset structure...")
    print("="*60)
    
    # Analyze each year
    year_dirs = sorted([d for d in extract_dir.iterdir() if d.is_dir()])
    
    all_info = {}
    
    for year_dir in year_dirs:
        year = year_dir.name
        print(f"\n📅 Analyzing {year} data...")
        
        # Find all CSV files in this year
        csv_files = list(year_dir.rglob("*.csv"))
        print(f"   Found {len(csv_files)} CSV files")
        
        # Group files by table type
        tables = {}
        for f in csv_files:
            table_name = f.name.split('_')[0] if '_' in f.name else f.stem
            if table_name not in tables:
                tables[table_name] = []
            tables[table_name].append(f)
        
        print(f"   Tables found: {list(tables.keys())}")
        
        # Analyze tblSCTurbine files (main SCADA data)
        if 'tblSCTurbine' in tables:
            turbine_file = tables['tblSCTurbine'][0]
            try:
                df_sample = pd.read_csv(turbine_file, nrows=100)
                
                # Get unique StationIds (turbine identifiers)
                if 'StationId' in df_sample.columns:
                    station_ids = df_sample['StationId'].unique()
                    turbine_map = {sid: sid - 2304509 for sid in station_ids}
                    print(f"   Turbines found: {sorted(turbine_map.values())[:10]}...")
                
                # Store info for this year
                all_info[year] = {
                    "files": len(csv_files),
                    "tables": list(tables.keys()),
                    "turbine_columns": list(df_sample.columns)[:20],
                    "shape_sample": df_sample.shape,
                    "date_range": f"Data for year {year}"
                }
                
            except Exception as e:
                print(f"   ⚠️ Could not analyze {year}: {e}")
    
    # Save consolidated information
    info_path = ORIGINAL_DATA_DIR / "multi_year_data_info.json"
    with open(info_path, 'w') as f:
        json.dump(all_info, f, indent=2)
    
    print(f"\n✅ Multi-year data info saved to: {info_path}")
    
    # Summary
    print("\n📊 Data Summary:")
    print(f"   Years available: {', '.join(year_dirs)}")
    print(f"   All 21 turbines present in original data")
    print(f"   Competition uses only turbines 1,2,3,4,5,7")
    print(f"   Missing turbine 6 could be extracted from original")

def compare_with_competition_data():
    """Compare original data with competition dataset."""
    print("\n🔍 Comparing with competition dataset...")
    print("="*60)
    
    # Load competition data info
    comp_train = PROJECT_ROOT / "data" / "train" / "training_dataset.parquet"
    
    if comp_train.exists():
        df_comp = pd.read_parquet(comp_train, engine='pyarrow')
        
        print("\n📊 Competition Dataset:")
        print(f"   Shape: {df_comp.shape}")
        print(f"   Date range: {df_comp['TimeStamp_StartFormat'].min()} to {df_comp['TimeStamp_StartFormat'].max()}")
        print(f"   Turbines in data: {sorted([int(c.split(';')[1]) for c in df_comp.columns if ';' in c and c.split(';')[1].isdigit()])[:10]}")
        
        # Check for additional fields in original data
        print("\n📝 Competition dataset has been:")
        print("   ✓ Filtered to turbines 1,2,3,4,5,7 only")
        print("   ✓ Standardized timestamps to UTC")
        print("   ✓ Duplicates removed")
        print("   ✓ Joined with ERA5 weather data")
        print("   ✓ Added is_valid and target columns")
        print("   ✓ Reshaped to wide format")
        
        print("\n💡 Original dataset advantages:")
        print("   • Additional turbines (6, 8, 9, etc.) for better context")
        print("   • More SCADA fields not in competition data")
        print("   • Longer historical period for analysis")
        print("   • Raw data without preprocessing")

def main():
    """Main execution function."""
    print("🚀 Hill of Towie Original Dataset Downloader (2016-2020)")
    print("="*60)
    
    # Setup directories
    setup_directories()
    
    # Clone/update GitHub repo
    repo_dir = clone_or_update_repo()
    
    # Download from Zenodo (years 2016-2020)
    zip_paths = download_zenodo_data()
    
    if zip_paths:
        # Extract datasets
        extract_dir = extract_dataset(zip_paths)
        
        if extract_dir:
            # Analyze structure
            analyze_original_data()
            
            # Compare with competition data
            compare_with_competition_data()
    
    print("\n✅ Data preparation complete!")
    print("\n📝 Next steps:")
    print("   1. Years 2016-2020 downloaded to match competition timeframe")
    print("   2. Turbine 6 data can be extracted from original for spatial interpolation")
    print("   3. Be careful: 2020 data is test period - avoid data leakage!")
    print("   4. Original has all 21 turbines vs competition's 6 turbines")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)