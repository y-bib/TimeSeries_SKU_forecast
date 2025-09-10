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

## Assumpition
- State space methods assumes Gaussian distribution of resuduals - did not tested that for the given data
- Outlnears are erros of input not results of promotions

## Notes
- **poisson_custome.py** also contains _Poisson regression_ and _State_Space model_ with Poisson rounding. The accuracy of forecast is approximately same.
- Primary analysis showed that there were no sufficient correlation  between numbers of sold SKUs, different rates of zeroes for SKUs: approximately 0, 0.3-0.35, 0.8
- also tried ARIMA but variation is not constant and mostly series donot pass stationarity test after first/second differences
- also tried to use Poisson HMM (Poisson Hidden Markov Model) with no success, need to try it in R
- The monthly average of the predictions tested on test data show relatively low MAE (need to show in a separate report), sufficiently better then ARIMA applied to monthly avarages 
