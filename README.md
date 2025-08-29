# Hill of Towie Wind Turbine Power Prediction

This project contains the solution for the Kaggle competition: [Hill of Towie Wind Turbine Power Prediction](https://www.kaggle.com/competitions/hill-of-towie-wind-turbine-power-prediction).

## 🎯 Competition Overview

**Goal**: Predict wind turbine power output based on operational and environmental data.

**Competition Type**: Community Competition  
**Deadline**: November 19, 2025  
**Evaluation**: TBD (check competition page for specific metric)

## 📁 Project Structure

```
hill-of-towie-wind-turbine/
├── data/
│   ├── raw/                    # Original data files
│   ├── processed/              # Cleaned and processed data
│   ├── train/                  # Training dataset
│   └── test/                   # Test dataset for submission
├── notebooks/                  # Jupyter notebooks for EDA and experiments
├── src/                        # Source code
├── models/                     # Trained model artifacts
├── submissions/                # Competition submissions
├── configs/                    # Configuration files
├── requirements.txt            # Python dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

1. **Clone and setup environment**:
   ```bash
   cd hill-of-towie-wind-turbine
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Set up Kaggle API** (if not already done):
   ```bash
   # Place your kaggle.json in ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. **Download data** (already downloaded):
   ```bash
   kaggle competitions download -c hill-of-towie-wind-turbine-power-prediction
   ```

4. **Start exploring**:
   ```bash
   jupyter lab notebooks/01_eda.ipynb
   ```

## 📊 Data Description

- **Training Data**: `data/train/training_dataset.parquet`
- **Test Data**: `data/test/submission_dataset.parquet`  
- **Sample Submission**: `data/raw/sample_model_submission.csv`

## 🔧 Usage

### Training a Model
```bash
python src/train.py --config configs/default.yaml
```

### Making Predictions
```bash
python src/predict.py --model models/best_model.pkl --output submissions/submission.csv
```

### Submitting to Kaggle
```bash
python src/submit.py --file submissions/submission.csv --message "Description of submission"
```

## 📈 Experiments and Results

Track your experiments in the `notebooks/` directory and save model artifacts in `models/`.

## 🏆 Competition Strategy

1. **Exploratory Data Analysis**: Understand the data patterns, seasonality, and correlations
2. **Feature Engineering**: Create relevant features for wind power prediction
3. **Model Selection**: Try various algorithms (XGBoost, LightGBM, Neural Networks)
4. **Cross Validation**: Implement proper time-series validation if applicable
5. **Ensemble Methods**: Combine multiple models for better performance

## 📝 Notes

- Wind power prediction typically involves weather data, turbine specifications, and operational parameters
- Consider seasonal patterns and time-based features
- Evaluate feature importance for model interpretability

## 🤝 Contributing

Feel free to experiment with different approaches and document your findings in the notebooks.

---
*Generated for the Hill of Towie Wind Turbine Power Prediction competition*