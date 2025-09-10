# Time Series Forecasting

## Overview
This project performs **time series forecasting** using XGBoost for daily demand prediction.

## Input
- File: `mock_test.csv`  
- Data type: **Poisson time series** (integer counts)  
- Features used: `day_of_week`, `month`, `season` (seasonality), and lag features  

## Output
- File: `mock_out.csv`  
- Forecast includes dates, SKU predictions and prediction intervals  

## Method
- **Model**: XGBoost Regressor with Poisson objective (`count:poisson`)  
- **Rounding**: Poisson rounding applied to ensure integer predictions  
- **Evaluation metrics**:
  - **MAE** (Mean Absolute Error)  
  - **MAPE** (Mean Absolute Percentage Error, biased-calculated)  
  - **Sample Variance** calculated on the test data  

## Notes
- Forecasting accounts for seasonality and lag effects.  
- Non-negative forecasts are guaranteed.  
- Prediction intervals are consistent with Poisson assumptions.
