# Time Series Forecasting

## Overview
This project performs **time series forecasting** using XGBoost for daily demand prediction.

## Input
- File: `mock_test.csv`  
- Intermittent Time Series/Poisson Time Series  

## Output
- File: `mock_out.csv`  
- Forecast includes dates, SKU predictions and  95%,97%,99% prediction intervals  

## Method
- **Outliners**: two round of zscore filter ()
- **Model**: XGBoost Regressor with seasonality and lags
- **Rounding**: Poisson rounding applied to ensure integer predictions  
- **Evaluation metrics**:
  - **MAE** (Mean Absolute Error)  
  - **MAPE** (Mean Absolute Percentage Error, non-zero values only, biased-calculated)  
  - **Sample Variance** calculated on the test data (20% of the whole data) 

## Notes
- poisson_custome contains Poisson regression and State_Space model with Poisson rounding. The accuracy of forecast is approximately same
