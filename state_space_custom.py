import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import statsmodels.api as sm # for poisson regression
import statsmodels.formula.api as smf

from statsmodels.tsa.statespace.structural import UnobservedComponents # for state space method
from errors_custom import errors


def state_space(df_cl, seasonality_col = 'Season'):
    df_tmp = pd.DataFrame({'y': df_cl['VALUE2'].values})
    y = df_tmp['y'].values  # original integer counts

    # 2) Fit a local level state-space model
    # Using "LocalLevel" component with Poisson-like log link

    #no seosonality
    #mod = UnobservedComponents(endog=y, level='local level', freq_seasonal=None)

    #  seasonality
    exog = df_cl[seasonality_col].values.reshape(-1, 1)
    mod = UnobservedComponents(endog=y, level='local level', exog=exog)
    #mod = UnobservedComponents(endog=y, level='local level', trend=True, seasonal=7, exog=exog)
    #mod = UnobservedComponents(endog=y, level='local level', seasonal=7,cycle=True, exog=exog)
    res = mod.fit(disp=False)


    # 3) Smoothed state estimates
    smoothed_state = res.smoothed_state[0]  # latent level

    # 4) Forecast (in-sample) expected counts
    pred_mean = res.predict()

    # 5) Convert predicted mean to integer counts via Poisson draws
    pred_integer = np.random.poisson(lam=np.maximum(pred_mean, 0))  # ensure non-negative

    # 6) Store in DataFrame
    df_tmp['pred_mean'] = pred_mean
    df_tmp['pred_integer'] = pred_integer

    # 7) Plot observed vs predicted
    plt.figure(figsize=(10,4))
    plt.plot(df_tmp['y'], label='Observed', marker='o')
    plt.plot(df_tmp['pred_mean'], label='Predicted mean', marker='x')
    plt.plot(df_tmp['pred_integer'], label='Predicted integer', marker='s', alpha=0.7)
    plt.xlabel("Time index")
    plt.ylabel("Counts")
    plt.title("State-Space Model Forecast for Counts")
    plt.legend()
    plt.show()

def fit_ucm_and_evaluate(df_cl, seasonality_col, test_size=0.2, plot=True):


    # 1) Prepare target and exogenous
    y = df_cl['VALUE2'].values
    exog = df_cl[seasonality_col].values.reshape(-1, 1)

    # Split into train/test
    n = len(y)
    split_idx = int((1 - test_size) * n)
    y_train, y_test = y[:split_idx], y[split_idx:]
    exog_train, exog_test = exog[:split_idx], exog[split_idx:]

    # 2) Fit local level model
    mod = UnobservedComponents(endog=y_train, level='local level', exog=exog_train)
    res = mod.fit(disp=False)

    # 3) Forecast on test set
    forecast_res = res.get_forecast(steps=len(y_test), exog=exog_test)
    pred_mean = forecast_res.predicted_mean
    pred_integer = np.random.poisson(lam=np.maximum(pred_mean, 0))

    err = errors(y_test, pred_mean, pred_integer)
    #print(err)

    # 5) Store results
    results = {
        "model": res,
        "y_train": y_train,
        "y_test": y_test,
        "pred_mean": pred_mean,
        "pred_integer": pred_integer,
        "mae_mean": err["mae_mean"],
        "mae_integer": err["mae_integer"],
        "mape_mean": err["mape_mean"],
        "mape_integer": err["mape_integer"],
        "var_test_sample": err["var_test_sample"]   # fixed key
    }


    # 6) Plot (train + test + forecast)
    if plot:
        plt.figure(figsize=(12, 5))
        plt.plot(y, label="Observed", marker='o')
        plt.axvline(split_idx, color="red", linestyle="--", label="Train/Test split")
        plt.plot(range(split_idx, n), pred_mean, label="Predicted mean", marker='x')
        plt.plot(range(split_idx, n), pred_integer, label="Predicted integer", marker='s', alpha=0.7)
        plt.xlabel("Time index")
        plt.ylabel("Counts")
        plt.title(f"State-Space Model Forecast (Test MAE MEAN = {err['mae_mean']:.3f},Test MAE INTEGER = {err['mae_integer']:.3f})")
        plt.legend()
        plt.show()

    return results

def state_space_out(df_cl, df_pred, n=30):

    y = df_cl['VALUE2'].values

    # exogs
    exog_cols = ['Month', 'Season', 'Day_of_Week']
    exog = df_cl[exog_cols].values if all(col in df_cl.columns for col in exog_cols) else None

    # Fit local-level state-space model
    mod = UnobservedComponents(endog=y, level='local level', exog=exog)
    res = mod.fit(disp=False)

    # Forecast future steps using exogenous values from df_pred
    exog_future = df_pred[exog_cols].values if exog is not None else None
    forecast_res = res.get_forecast(steps=n, exog=exog_future)

    # Predicted mean (float)
    pred_mean = forecast_res.predicted_mean

    # Convert to integers via Poisson sampling
    pred_integer = np.random.poisson(lam=np.maximum(pred_mean, 0))

    # Add forecasts to df_pred
    df_pred = df_pred.copy()
    df_pred['FORECAST'] = pred_integer

    return df_pred
