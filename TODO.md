# Hill of Towie Wind Turbine Power Prediction - Project TODO

## 🎯 Competition Goal
Predict power output for Turbine 1 using data from nearby turbines (2, 3, 4, 5, 7) to minimize Mean Absolute Error (MAE) for turbine upgrade measurement scenarios.

## 📋 Critical Fixes Needed (Priority 1)

### ❌ Current Issues
- [ ] **Wrong target column**: Currently using `wtc_PowerRef_endvalue;1` instead of `target`
- [ ] **Missing is_valid handling**: Not filtering data for normal operation periods
- [ ] **Generic EDA**: Not tailored to wind turbine domain
- [ ] **Incorrect feature selection**: Not excluding Turbine 1 features from training

### ✅ Immediate Actions
- [ ] Fix target column in all notebooks and scripts
- [ ] Implement is_valid filtering in training pipeline
- [ ] Update data loading to properly separate features and target
- [ ] Verify submission format matches competition requirements

## 🔍 Data Understanding & EDA (Priority 2)

### 1. Data Validation
- [ ] Verify data structure matches competition description
- [ ] Confirm training data: 2016-2019 for all turbines
- [ ] Confirm test data: 2020 for turbines 2,3,4,5,7 only
- [ ] Check target column is Turbine 1 active power (clipped at 0)
- [ ] Validate is_valid flag correlates with ShutdownDuration=0 and wtc_ScReToOp_timeon=600

### 2. Wind Turbine Specific Analysis
- [ ] **Turbine Layout Visualization**
  - [ ] Create spatial plot of turbine positions (if coordinates available)
  - [ ] Identify prevailing wind directions
  - [ ] Map potential wake interaction zones

- [ ] **Inter-Turbine Correlations**
  - [ ] Power correlation matrix between all turbines
  - [ ] Time-lagged correlations for wake effects
  - [ ] Correlation variation by wind direction

- [ ] **Power Curve Analysis**
  - [ ] Plot power vs wind speed for each turbine
  - [ ] Identify cut-in, rated, and cut-out wind speeds
  - [ ] Compare power curves across turbines
  - [ ] Detect anomalies and underperformance

- [ ] **Operational State Analysis**
  - [ ] Availability patterns for each turbine
  - [ ] Shutdown duration distributions
  - [ ] Correlation between shutdowns across turbines

### 3. Weather Pattern Analysis (ERA5 Data)
- [ ] **Wind Analysis**
  - [ ] Wind roses for 10m and 100m heights
  - [ ] Wind shear profiles (100m/10m ratio)
  - [ ] Seasonal and diurnal wind patterns
  - [ ] Extreme wind event identification

- [ ] **Atmospheric Conditions**
  - [ ] Temperature effects on air density and power
  - [ ] Pressure patterns and frontal systems
  - [ ] Cloud cover impact on atmospheric stability
  - [ ] Precipitation effects on performance

- [ ] **Directional Analysis**
  - [ ] Power output by wind direction sectors
  - [ ] Identify wake-affected sectors
  - [ ] Seasonal variation in wind direction

## 🛠️ Feature Engineering (Priority 3)

### 1. Wind-Related Features
- [ ] **Wind Shear**
  ```python
  wind_shear = ERA5_wind_speed_100m / ERA5_wind_speed_10m
  ```
- [ ] **Wind Vector Components**
  ```python
  u_component = wind_speed * cos(wind_direction)
  v_component = wind_speed * sin(wind_direction)
  ```
- [ ] **Turbulence Intensity**
  ```python
  TI = wind_speed_std / wind_speed_mean
  ```

### 2. Turbine Interaction Features
- [ ] **Wake Indicators**
  - [ ] Upstream turbine operating status
  - [ ] Wind direction alignment between turbines
  - [ ] Power deficit ratios

- [ ] **Neighboring Turbine Features**
  - [ ] Mean power of nearby turbines
  - [ ] Power variance across farm
  - [ ] Weighted average by distance/direction

### 3. Temporal Features
- [ ] **Lag Features**
  - [ ] Power lags: 10min, 30min, 1hr, 3hr, 6hr
  - [ ] Wind speed/direction lags
  - [ ] Multi-turbine lagged features

- [ ] **Rolling Statistics**
  - [ ] Rolling mean/std/min/max (1hr, 3hr, 6hr, 24hr windows)
  - [ ] Exponential weighted moving averages

- [ ] **Cyclical Encodings**
  ```python
  hour_sin = sin(2π * hour / 24)
  hour_cos = cos(2π * hour / 24)
  month_sin = sin(2π * month / 12)
  month_cos = cos(2π * month / 12)
  ```

### 4. Atmospheric Stability Features
- [ ] Richardson number approximation
- [ ] Day/night stability indicators
- [ ] Seasonal atmospheric patterns

### 5. Operational Features
- [ ] Time since last shutdown
- [ ] Cumulative operational hours
- [ ] Maintenance period indicators
- [ ] Capacity factor calculations

## 🤖 Modeling Strategy (Priority 4)

### 1. Baseline Models
- [ ] **Physics-Based Baseline**
  - [ ] Power curve interpolation
  - [ ] IEC standard power curve model
  - [ ] Wake loss models (Jensen/Frandsen)

- [ ] **Statistical Baseline**
  - [ ] Historical average by conditions
  - [ ] Nearest neighbor turbine matching
  - [ ] Simple linear regression

### 2. Machine Learning Models
- [ ] **Tree-Based Models**
  - [ ] XGBoost with custom objective for MAE
  - [ ] LightGBM with time-aware features
  - [ ] CatBoost for categorical encodings
  - [ ] Random Forest for feature importance

- [ ] **Time Series Models**
  - [ ] LSTM/GRU for sequential patterns
  - [ ] Temporal Fusion Transformer
  - [ ] Prophet for seasonal decomposition
  - [ ] ARIMA variants for residual modeling

- [ ] **Ensemble Approaches**
  - [ ] Weighted average by performance
  - [ ] Stacking with meta-learner
  - [ ] Blending physics and ML models

### 3. Model Configuration
```python
# Example XGBoost configuration for MAE
params = {
    'objective': 'reg:absoluteerror',  # MAE objective
    'eval_metric': 'mae',
    'max_depth': 8,
    'learning_rate': 0.05,
    'n_estimators': 2000,
    'early_stopping_rounds': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'gamma': 0.1
}
```

## 📊 Validation Strategy (Priority 5)

### 1. Time Series Cross-Validation
- [ ] **Blocked Time Series Split**
  ```python
  # Example: 3-month training, 1-month gap, 1-month validation
  splits = [
      (train_2016_Q1, val_2016_Q3),
      (train_2016_Q1Q2, val_2016_Q4),
      (train_2016, val_2017_Q1),
      ...
  ]
  ```

- [ ] **Seasonal Validation**
  - [ ] Train on 3 years, validate on same season in 4th year
  - [ ] Ensures model handles seasonal patterns

- [ ] **Turbine Upgrade Simulation**
  - [ ] Hide Turbine 1 data for specific periods
  - [ ] Mimic real upgrade measurement scenarios

### 2. Evaluation Metrics
- [ ] **Primary**: Mean Absolute Error (MAE)
- [ ] **Secondary Metrics**:
  - [ ] RMSE for outlier sensitivity
  - [ ] MAPE for relative errors
  - [ ] Quantile losses for uncertainty

- [ ] **Stratified Analysis**:
  - [ ] Error by power bins (low/medium/high)
  - [ ] Error by wind speed ranges
  - [ ] Error by time of day/season
  - [ ] Error by operational state

## 📁 Project Structure Updates

### 1. Directory Organization
```
hill-of-towie-wind-turbine/
├── data/
│   ├── raw/                 # Original competition data
│   ├── processed/           # Cleaned, is_valid filtered
│   ├── features/            # Engineered features
│   ├── external/            # Additional weather data
│   └── interim/             # Intermediate processing
├── notebooks/
│   ├── 01_data_validation.ipynb     # Verify data integrity
│   ├── 02_turbine_eda.ipynb        # Turbine-specific analysis
│   ├── 03_weather_analysis.ipynb    # ERA5 patterns
│   ├── 04_feature_engineering.ipynb # Feature creation
│   ├── 05_baseline_models.ipynb     # Simple models
│   ├── 06_ml_experiments.ipynb      # Advanced models
│   └── 07_ensemble.ipynb            # Model combination
├── src/
│   ├── data/
│   │   ├── load_data.py            # Data loading utilities
│   │   ├── validate_data.py        # Data quality checks
│   │   └── make_dataset.py         # Dataset preparation
│   ├── features/
│   │   ├── wind_features.py        # Wind-related features
│   │   ├── turbine_features.py     # Turbine interactions
│   │   ├── temporal_features.py    # Time-based features
│   │   └── build_features.py       # Feature pipeline
│   ├── models/
│   │   ├── baseline.py             # Baseline models
│   │   ├── tree_models.py          # XGB, LGBM, etc.
│   │   ├── neural_models.py        # LSTM, GRU, etc.
│   │   └── ensemble.py             # Model combination
│   ├── validation/
│   │   ├── time_series_cv.py       # CV strategies
│   │   └── metrics.py              # Evaluation metrics
│   └── visualization/
│       ├── turbine_plots.py        # Turbine visualizations
│       ├── weather_plots.py        # Weather analysis plots
│       └── model_plots.py          # Model performance plots
├── configs/
│   ├── data_config.yaml            # Data processing settings
│   ├── feature_config.yaml         # Feature engineering settings
│   └── model_config.yaml           # Model hyperparameters
└── scripts/
    ├── run_pipeline.py              # Full pipeline execution
    ├── train_model.py               # Model training
    ├── make_submission.py           # Generate predictions
    └── validate_submission.py       # Check submission format
```

### 2. Configuration Files
- [ ] Create YAML configs for reproducibility
- [ ] Document all hyperparameters
- [ ] Version control experiments

## 🚀 Implementation Timeline

### Week 1: Data Foundation
- [ ] Fix critical issues (wrong target, is_valid)
- [ ] Complete data validation
- [ ] Create corrected EDA notebooks

### Week 2: Domain Analysis
- [ ] Turbine correlation analysis
- [ ] Weather pattern analysis
- [ ] Power curve modeling

### Week 3: Feature Engineering
- [ ] Implement wind features
- [ ] Create turbine interaction features
- [ ] Build temporal features

### Week 4: Baseline Models
- [ ] Physics-based baseline
- [ ] Simple ML models
- [ ] Establish benchmark scores

### Week 5-6: Advanced Modeling
- [ ] Optimize tree-based models
- [ ] Experiment with neural networks
- [ ] Develop ensemble strategies

### Week 7: Validation & Tuning
- [ ] Comprehensive cross-validation
- [ ] Hyperparameter optimization
- [ ] Error analysis

### Week 8: Finalization
- [ ] Final model selection
- [ ] Documentation
- [ ] Submission preparation

## 📝 Key Insights & Lessons Learned

### Competition-Specific Insights
- Turbine 1 is the target (we don't have its 2020 data)
- Use only turbines 2,3,4,5,7 features for prediction
- is_valid flag is crucial for proper scoring
- Focus on MAE, not RMSE

### Wind Turbine Domain Knowledge
- Wake effects are directional and distance-dependent
- Power curves are non-linear with hysteresis
- Atmospheric stability affects performance
- Icing and soiling can impact efficiency

### Modeling Insights
- [ ] Document what works and what doesn't
- [ ] Track feature importance across models
- [ ] Note seasonal performance variations
- [ ] Record computational requirements

## 🎯 Success Criteria

1. **Correct Implementation**
   - ✅ Using correct target column
   - ✅ Properly filtering with is_valid
   - ✅ Not using Turbine 1 features in training

2. **Comprehensive Analysis**
   - ✅ Understanding turbine interactions
   - ✅ Capturing weather impacts
   - ✅ Identifying temporal patterns

3. **Robust Modeling**
   - ✅ Beating baseline by >20%
   - ✅ Stable cross-validation scores
   - ✅ Good performance across all conditions

4. **Competition Ready**
   - ✅ Correct submission format
   - ✅ Reproducible pipeline
   - ✅ Well-documented approach

## 🔗 Resources

- [Competition Page](https://www.kaggle.com/competitions/hill-of-towie-wind-turbine-power-prediction)
- [Getting Started Notebook](https://www.kaggle.com/code/gabecalvo/hill-of-towie-power-prediction-getting-started)
- [Original Dataset](https://github.com/resgroup/hill-of-towie-open-source-analysis)
- [ERA5 Documentation](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5)
- [Wind Turbine Wake Models](https://www.wind-energ-sci.net/)

---
*Last Updated: [Current Date]*
*Competition Deadline: November 19, 2025*