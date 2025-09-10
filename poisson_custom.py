#!/usr/bin/env python
# coding: utf-8

# In[1]:


from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import statsmodels.api as sm # for poisson regression
import statsmodels.formula.api as smf

from statsmodels.tsa.statespace.structural import UnobservedComponents # for state space method

import xgboost as xgb #for xboost

import os





# ## functions

# In[4]:

# returns data frame for one SKU only
def sku_extract(df, SKU = '991234-A'):
    df_cl = df[['DATE','VALUE','ITEM_CODE']][df.ITEM_CODE == SKU].copy()
    df_cl.reset_index(drop=True, inplace=True)
    return df_cl


# In[5]:

# point plot
def plot_SKU(df_cl, y_col='VALUE', title='Sold per day'):
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df_cl, x=df_cl.index, y=df_cl[y_col])
    # plt.ylim(0, 5)  
    plt.title(title)
    plt.ylabel('Sold')
    plt.xlabel('Days')
    plt.grid(True)
    plt.tight_layout()
    plt.show()






#two options to fix outliners
###########outliners via zscore
# def zscore(s, window, thresh=3, return_all=False):

#     def apply_once(series, window=30, thresh=3, return_all=False):
#         roll = series.rolling(window=window, min_periods=1, center=True)
#         avg = roll.mean()
#         std = roll.std(ddof=0)
#         z = series.sub(avg).div(std)  
#         m = z < thresh
        
#         # Replace outliers with rolling mean, clipped to at least 1
#         replaced = series.where(m, avg).round().clip(lower=1).astype(int)    
#         return replaced, z, avg, std, m

#     # First pass
#     s1, z1, avg1, std1, m1 = apply_once(s, window=window, thresh=thresh)

#     # Second pass
#     s2, z2, avg2, std2, m2 = apply_once(s1, window=window, thresh=thresh)

#     if return_all:
#         return avg2, m1 & m2
#     return s2

# two rounds of rolling average check and case for many zeroes
def zscore(s, window, thresh=2, return_all=False):

    def apply_once(series):
        roll = series.rolling(window=window, min_periods=1, center=True)
        avg = roll.mean()
        std = roll.std(ddof=0)
        z = series.sub(avg).div(std)  
        m = z.between(-thresh, thresh)
        return series.where(m, avg), z, avg, std, m

    # If >50% zeros → clip at Q95 instead of z-score
    zero_ratio = (s == 0).mean()
    if zero_ratio > 0.75:
        q95 = s.quantile(0.95)
        s_clipped = s.clip(upper=q95).astype(int)
        mask = s <= q95  # True if unchanged, False if clipped
        if return_all:
            return s_clipped, mask
        return s_clipped

    # Otherwise apply two-pass z-score method
    s1, z1, avg1, std1, m1 = apply_once(s)
    #s2, z2, avg2, std2, m2 = apply_once(s1)

    if return_all:
        return avg1, m1
    return s1


def plot_zscore(df_cl, avg, m):

    plt.figure(figsize=(10, 5))

    # Original data
    df_cl['VALUE'].plot(label='Original Data', color='black', alpha=0.7)

    # Rolling mean
    avg.plot(label='Rolling Mean', color='blue', lw=2)

    # Outliers
    df_cl.loc[~m, 'VALUE'].plot(
        label='Outliers', marker='o', ls='', color='red', markersize=6
    )

    # Replacement values
    df_cl.loc[~m, 'VALUE2'].plot(
        label='Replacements', marker='x', ls='', color='green', markersize=6
    )

    plt.xlabel("Date / Index")
    plt.ylabel("VALUE")
    plt.title("Z-Score Outlier Detection and Replacement")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


######### outliner cap above threshold
def cap_outliers(df, col='VALUE', outl=10): # replaces outlines (above outl level) with maximum
    df_copy = df.copy()
    mask_valid = df_copy[col] <= outl
    max_valid = df_copy.loc[mask_valid, col].max()
    df_copy.loc[df_copy[col] > outl, col] = max_valid
    return df_copy

def plot_cap_outliers(VAL, VAL2, outl=10):
    mask_outliers = VAL > outl
    plt.figure(figsize=(12, 5))
    ax = plt.subplot()
    VAL.plot(label='Original data', ax=ax)

    # Plot outliers as red circles
    VAL.loc[mask_outliers].plot(
        label='Outliers', marker='o', ls='', color='red', ax=ax
    )
    # Plot the replaced values as green crosses
    VAL2.loc[mask_outliers].plot(
        label='Replaced', marker='x', ls='', color='green', ax=ax
    )

    # #Plot a horizontal line at the max of the valid values
    # max_valid = VAL.loc[~mask_outliers].max()
    # plt.axhline(max_valid, color='blue', linestyle='--', label='Max of valid values')
    plt.title('Original data with outliers and replaced values')
    plt.xlabel('Index')
    plt.ylabel('VALUE')
    plt.legend()
    plt.show()


# get season based on month number
def get_season(month):
    if month in [6, 7, 8]:
        return 0
    elif month in [9, 10, 11]:
        return 1
    elif month in [12, 1, 2]:
        return 2
    elif month in [3, 4, 5]:
        return 3
    else:
        return np.nan


def date_split(df_cl):
    df_cl['DATE'] = pd.to_datetime(df_cl['DATE'])
    df_cl['Year'] = df_cl['DATE'].dt.year
    df_cl['Month'] = df_cl['DATE'].dt.month
    df_cl['Day'] = df_cl['DATE'].dt.day
    df_cl['Day_of_Week'] = df_cl['DATE'].dt.dayofweek
    df_cl['Season'] = df_cl['Month'].apply(get_season)







def means_and_cor(df_cl):

    daily_means = df_cl.groupby('Day_of_Week')['VALUE2'].mean().reset_index()
    daily_means.columns = ['Day_of_Week', 'Mean_Value']

    print("--- Average 'VALUE2' per Day of Week ---")
    print(daily_means)

    # Calculate correlation for Day of Week

    correlation = daily_means['Day_of_Week'].corr(daily_means['Mean_Value'])
    print(f"\n--- Pearson Correlation Coefficient ---")
    print(f"Correlation between Day of Week and Mean Value: {correlation:.4f}")

    correlation = df_cl['VALUE2'].corr(df_cl['Day_of_Week'])
    print("Correlation between VALUE2 and Day_of_Week:", correlation)

    # Group by Month and calculate mean
    monthly_means = df_cl.groupby('Month')['VALUE2'].mean().reset_index()
    monthly_means.columns = ['Month', 'Mean_Value']

    print("\n--- Average 'VALUE2' per Month ---")
    print(monthly_means)

    # Calculate correlation for Month

    correlation = monthly_means['Month'].corr(monthly_means['Mean_Value'])
    print(f"\n--- Pearson Correlation Coefficient ---")
    print(f"Correlation between Month and Mean Value: {correlation:.4f}")

    correlation = df_cl['VALUE2'].corr(df_cl['Month'])
    print("Correlation between VALUE2 and Month:", correlation)


    seasonal_means = df_cl.groupby('Season')['VALUE2'].mean().reset_index()
    seasonal_means.columns = ['Season', 'Mean_Value']

    print("--- Average 'VALUE2' per Season ---")
    print(seasonal_means)
    correlation = seasonal_means['Season'].corr(seasonal_means['Mean_Value'])

    print("\n--- Pearson Correlation Coefficient ---")
    print(f"Correlation between Season and Mean Value: {correlation:.4f}")

    correlation = df_cl['VALUE2'].corr(df_cl['Season'])
    print("Correlation between VALUE2 and Season:", correlation)





def poisson_regression(df_cl):
    # 1) Make sure the DataFrame name is consistent
    df_tmp = pd.DataFrame({'y': df_cl['VALUE2'].values})

    # 2) Create lagged column and day index BEFORE fitting
    df_tmp['lag1'] = df_tmp['y'].shift(30).fillna(0)  # lag1
    df_tmp['day'] = range(len(df_tmp))               # day index

    # 3) Fit Poisson regression
    model = smf.glm('y ~ lag1 + day', data=df_tmp, family=sm.families.Poisson()).fit()

    # 4) Predict and store predictions
    df_tmp['predicted'] = model.predict(df_tmp)

    # 5) Optional: print summary
    print(model.summary())

    plt.figure(figsize=(10, 4))

    # Plot observed counts
    plt.plot(df_tmp['y'], label='Observed', marker='o', linestyle='-', alpha=0.7)

    # Plot predicted counts
    plt.plot(df_tmp['predicted'], label='Predicted', marker='x', linestyle='--', alpha=0.9)

    plt.xlabel("Time index")
    plt.ylabel("Counts")
    plt.title("Observed vs Predicted Counts (Poisson Regression)")
    plt.legend()
    plt.show()
    return df_tmp[['predicted']]





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


def errors(y_test, pred_mean, pred_integer):
    # y_test = np.asarray(y_test)
    # pred_mean = np.asarray(pred_mean)
    # pred_integer = np.asarray(pred_integer)

    n = len(y_test)

    # MAE
    mae_mean = mean_absolute_error(y_test, pred_mean)
    mae_integer = mean_absolute_error(y_test, pred_integer)

    # Test sample variance (using integer predictions)
    #var_test_sample = (1 / (n - 1)) * np.sum((y_test - pred_integer) ** 2)
    residuals = y_test - pred_integer
    mean_res = np.mean(residuals)
    var_test_sample = (1 / (n - 1)) * np.sum((residuals - mean_res) ** 2)

    # MAPE (byised, ignores zeros)
    mask = y_test != 0
    mape_mean = np.mean(np.abs((y_test[mask] - pred_mean[mask]) / y_test[mask]))*100
    mape_integer = np.mean(np.abs((y_test[mask] - pred_integer[mask]) / y_test[mask]))*100

    return {
        "mae_mean": mae_mean,
        "mae_integer": mae_integer,
        "mape_mean": mape_mean,
        "mape_integer": mape_integer,
        "var_test_sample": var_test_sample
    }


from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.statespace.structural import UnobservedComponents
from sklearn.metrics import mean_absolute_percentage_error


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



def fit_xgb_poisson(df_cl, test_size=0.2, max_lag=3, plot=True):

    # 1) Prepare target and lag features
    df_tmp = pd.DataFrame({'y': df_cl['VALUE2'].values})
    for lag in range(1, max_lag + 1):
        df_tmp[f'lag{lag}'] = df_tmp['y'].shift(lag)

    # Drop initial rows with NaNs from lagging
    df_tmp = df_tmp.dropna().reset_index(drop=True)

    # Add calendar features
    df_tmp['Day'] = df_cl['Day'].iloc[max_lag:].values
    df_tmp['Day_of_Week'] = df_cl['Day_of_Week'].iloc[max_lag:].values

    # Train/test split
    train_size = int(len(df_tmp) * (1 - test_size))
    train = df_tmp.iloc[:train_size]
    test = df_tmp.iloc[train_size:]

    X_train = train.drop(columns=['y'])
    y_train = train['y']
    X_test = test.drop(columns=['y'])
    y_test = test['y']

    # 2) Define and fit XGBoost Poisson model
    model = xgb.XGBRegressor(
        objective='count:poisson',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 3) Predict
    pred_mean = model.predict(X_test)
    pred_integer = np.round(pred_mean).astype(int)
    pred_integer[pred_integer < 0] = 0  # clip negatives

    err = errors(y_test, pred_mean, pred_integer)
    #print(err)

    # 5) Store results
    results = {
        "model": model,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "pred_mean": pred_mean,
        "pred_integer": pred_integer,
        "mae_mean": err["mae_mean"],
        "mae_integer": err["mae_integer"],
        "mape_mean": err["mape_mean"],
        "mape_integer": err["mape_integer"],
        "var_test_sample": err["var_test_sample"]  
    }

    # 6) Plot observed vs predicted
    if plot:
        y_full = np.concatenate([y_train, y_test])
        plt.figure(figsize=(12, 5))
        plt.plot(y_full, label="Observed", marker='o')
        plt.axvline(train_size, color="red", linestyle="--", label="Train/Test split")
        plt.plot(range(train_size, len(y_full)), pred_mean, label="Predicted mean", marker='x')
        plt.plot(range(train_size, len(y_full)), pred_integer, label="Predicted integer", marker='s', alpha=0.7)
        plt.xlabel("Time index")
        plt.ylabel("Counts")
        plt.title(f"XGBoost Poisson Forecast\nMAE Mean = {err['mae_mean']:.3f}, MAE Integer = {err['mae_integer']:.3f}")
        plt.legend()
        plt.show()

    return results



#Xboost with all seasonality and lags
def fit_xgb_poisson2(df_cl, test_size=0.2, max_lag=3, plot=True):
    # -------------------------
    # 1) Prepare data
    # -------------------------
    df_tmp = pd.DataFrame({'y': df_cl['VALUE2'].values})

    # Create lag features
    for lag in range(1, max_lag + 1):
        df_tmp[f'lag{lag}'] = df_tmp['y'].shift(lag)

    # Drop initial rows with NaNs
    df_tmp = df_tmp.dropna().reset_index(drop=True)

    # Add seasonal features
    df_tmp['Month'] = df_cl['Month'].iloc[max_lag:].values
    df_tmp['Season'] = df_cl['Season'].iloc[max_lag:].values
    df_tmp['Day_of_Week'] = df_cl['Day_of_Week'].iloc[max_lag:].values

    # -------------------------
    # 2) Train/test split
    # -------------------------
    train_size = int(len(df_tmp) * (1 - test_size))
    train = df_tmp.iloc[:train_size]
    test = df_tmp.iloc[train_size:]

    X_train = train.drop(columns=['y'])
    y_train = train['y']
    X_test = test.drop(columns=['y'])
    y_test = test['y']

    # -------------------------
    # 3) Train XGBoost Poisson model
    # -------------------------
    model = xgb.XGBRegressor(
        objective='count:poisson',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    # -------------------------
    # 4) Predict counts
    # -------------------------
    pred_mean = model.predict(X_test)
    pred_integer = np.round(pred_mean).astype(int)
    pred_integer[pred_integer < 0] = 0

    # -------------------------
    # 5) Calculate errors
    # -------------------------
    err = errors(y_test, pred_mean, pred_integer)

    results = {
        "model": model,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "pred_mean": pred_mean,
        "pred_integer": pred_integer,
        "mae_mean": err["mae_mean"],
        "mae_integer": err["mae_integer"],
        "mape_mean": err["mape_mean"],
        "mape_integer": err["mape_integer"],
        "var_test_sample": err["var_test_sample"]
    }

    # -------------------------
    # 6) Plot observed vs predicted
    # -------------------------
    if plot:
        y_full = np.concatenate([y_train, y_test])
        plt.figure(figsize=(12, 5))
        plt.plot(y_full, label="Observed", marker='o')
        plt.axvline(train_size, color="red", linestyle="--", label="Train/Test split")
        plt.plot(range(train_size, len(y_full)), pred_mean, label="Predicted mean", marker='x')
        plt.plot(range(train_size, len(y_full)), pred_integer, label="Predicted integer", marker='s', alpha=0.7)
        plt.xlabel("Time index")
        plt.ylabel("Counts")
        plt.title(f"XGBoost Poisson Forecast (Poisson2)")
        plt.legend()
        plt.show()

    return results


# In[24]:


# results = fit_xgb_poisson2(df_cl, test_size=0.2, max_lag=3)
# print(f"MAE (mean): {results['mae_mean']:.3f}")
# print(f"MAE (integer): {results['mae_integer']:.3f}")
# print(f"MAPE (%) mean: {results['mape_mean']:.2f}%")
# print(f"MAPE (%) integer: {results['mape_integer']:.2f}%")


# In[25]:


import pandas as pd

def generate_calendar(start_date, end_date):

    # Create date range
    dates = pd.date_range(start=pd.to_datetime(start_date), end=pd.to_datetime(end_date), freq='D')

    # Build DataFrame
    df = pd.DataFrame({"Date": dates})
    df["Day"] = df["Date"].dt.day
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year
    df["Day_of_Week"] = df["Date"].dt.dayofweek  

    df["Season"] = df["Month"].apply(get_season)

    return df


# In[26]:


import xgboost as xgb

def fit_xgb_poisson_out(df_cl, df_pred, max_lag=3, n=30):


    df_tmp = pd.DataFrame({'y': df_cl['VALUE2'].values})
    for lag in range(1, max_lag + 1):
        df_tmp[f'lag{lag}'] = df_tmp['y'].shift(lag)

    df_tmp = df_tmp.dropna().reset_index(drop=True)

    # calendar added
    df_tmp['Month'] = df_cl['Month'].iloc[max_lag:].values
    df_tmp['Season'] = df_cl['Season'].iloc[max_lag:].values
    df_tmp['Day_of_Week'] = df_cl['Day_of_Week'].iloc[max_lag:].values

    # 2) Train XGBoost Poisson model
    X_train = df_tmp.drop(columns=['y'])
    y_train = df_tmp['y']

    model = xgb.XGBRegressor(
        objective='count:poisson',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 3) Forecast next n steps using recursive lag updates
    preds_mean = []
    preds_int = []
    last_known = df_tmp.iloc[-1:].copy()

    for i in range(n):
        X_input = last_known.drop(columns=['y']).copy()
        pred_mean = model.predict(X_input)[0]
        preds_mean.append(pred_mean)

        # Convert to integer via Poisson sampling
        pred_int = np.random.poisson(lam=np.maximum(pred_mean, 0))
        preds_int.append(pred_int)

        # Update lag features for next step
        new_row = {}
        new_row['y'] = pred_int  # use integer for recursion
        for lag in range(1, max_lag + 1):
            if lag == 1:
                new_row[f'lag{lag}'] = pred_int
            else:
                new_row[f'lag{lag}'] = last_known[f'lag{lag-1}'].values[0]

        # Copy calendar features from df_pred
        for col in ['Month', 'Season', 'Day_of_Week']:
            new_row[col] = df_pred.iloc[i][col]

        last_known = pd.DataFrame([new_row])

    # 4) Add forecasts to df_pred
    df_pred = df_pred.copy()
    df_pred['FORECAST'] = preds_int  # integer forecasts

    return df_pred


# In[27]:


def fit_xgb_poisson2_out(df_cl, df_pred, max_lag=3, n=30):

    # 1) Prepare lag features from historical data
    df_tmp = pd.DataFrame({'y': df_cl['VALUE2'].values})
    for lag in range(1, max_lag + 1):
        df_tmp[f'lag{lag}'] = df_tmp['y'].shift(lag)
    df_tmp = df_tmp.dropna().reset_index(drop=True)

    # Add seasonal features from historical data
    df_tmp['Month'] = df_cl['Month'].iloc[max_lag:].values
    df_tmp['Season'] = df_cl['Season'].iloc[max_lag:].values
    df_tmp['Day_of_Week'] = df_cl['Day_of_Week'].iloc[max_lag:].values

    # 2) Train XGBoost Poisson model
    X_train = df_tmp.drop(columns=['y'])
    y_train = df_tmp['y']

    model = xgb.XGBRegressor(
        objective='count:poisson',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 3) Forecast next n days using recursive lag updates
    preds_int = []
    last_known = df_tmp.iloc[-1:].copy()

    for i in range(n):
        X_input = last_known.drop(columns=['y']).copy()
        pred_mean = model.predict(X_input)[0]

        # Integer forecast via Poisson sampling
        pred_int = np.random.poisson(lam=np.maximum(pred_mean, 0))
        preds_int.append(pred_int)

        # Update lag features for next iteration
        new_row = {'y': pred_int}
        for lag in range(1, max_lag + 1):
            if lag == 1:
                new_row[f'lag{lag}'] = pred_int
            else:
                new_row[f'lag{lag}'] = last_known[f'lag{lag-1}'].values[0]

        # Copy seasonal features from df_pred
        for col in ['Month', 'Season', 'Day_of_Week']:
            new_row[col] = df_pred.iloc[i][col]

        last_known = pd.DataFrame([new_row])

    # 4) Add forecasts to df_pred
    df_pred = df_pred.copy()
    df_pred['FORECAST'] = preds_int

    return df_pred


# In[28]:


import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.structural import UnobservedComponents

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


# In[29]:


def pred_out(df_pred, SKU, variance):
    df_out = df_pred.copy()
    df_out['ITEM_CODE'] = SKU
    n=len(df_pred.index)
    # Standard error from variance
    std = np.sqrt(variance)

    # Z-scores for intervals
    z_scores = {
        95: 1.96,
        97: 2.17,   
        99: 2.58
    }

    for level, z in z_scores.items():
        df_out[f"LOWER_{level}"] = df_out["FORECAST"] - z * std
        df_out[f"UPPER_{level}"] = df_out["FORECAST"] + z * std

    df_out = df_out[['Date', 'ITEM_CODE','FORECAST', 'LOWER_95', 'UPPER_95', 'LOWER_97', 'UPPER_97', 'LOWER_99', 'UPPER_99']]
    return df_out


# In[30]:


def process_sku(df, SKU):
    print(f"Processing SKU: {SKU}")

    # 1) Extract SKU data
    df_cl = sku_extract(df, SKU)

    # 2) Compute z-score for VALUE
    df_cl['VALUE2'] = zscore(df_cl['VALUE'], window=90)
    avg, m = zscore(df_cl['VALUE'], window=90, return_all=True)

    plot_zscore(df_cl, avg, m)


    # outl=20
    # df_cl['VALUE2'] = cap_outliers(df_cl, col='VALUE1', outl=outl)['VALUE']
    # plot_cap_outliers(df_cl['VALUE1'],df_cl['VALUE2'],outl)

    # 3) Data prep

    date_split(df_cl)
    plot_SKU(df_cl, y_col='VALUE2', title='Sold per day, no outliers')

    df_cl['VALUE2'].rolling(window=7).mean().plot(
    lw=2, color='blue', label='7-day Rolling Mean')


    # mean for weekdays, month and season
    means_and_cor(df_cl)

    # 4) Fit model and evaluate
    #results = fit_ucm_and_evaluate(df_cl, seasonality_col='Day_of_Week', test_size=0.2, plot=True)
    results = fit_xgb_poisson2(df_cl, test_size=0.2, max_lag=7, plot=True)  # other good option space_state function

    # 5) Print evaluation metrics
    print("Test MAE (mean):", results["mae_mean"])
    print("Test MAE (integer):", results["mae_integer"])
    print("Test MAPE (mean):", results["mape_mean"])
    print("Test MAPE (integer):", results["mape_integer"])
    print("Sample (test) variance:", results["var_test_sample"])
    var1 = results["var_test_sample"]


    # 6) Generate forecast calendar
    df_pred = generate_calendar('2025-09-01','2025-09-30')

    # 7) Fit XGBoost Poisson and get forecasts
    df_pred = fit_xgb_poisson2_out(df_cl, df_pred, max_lag=7, n=30)

    # 8) Generate final prediction dataframe with intervals
    df_out = pred_out(df_pred, SKU, var1)

    # 9) Plot forecasts with intervals
    plot_forecast_with_intervals(df_out)

    # 10) Save or append to CSV
    filename_out = f"data/mock_out.csv"
    if os.path.exists(filename_out):
        df_out.to_csv(filename_out, mode='a', index=False, header=False)
    else:
        df_out.to_csv(filename_out, mode='w', index=False, header=True)

    return df_out


# In[31]:


import matplotlib.pyplot as plt

def plot_forecast_with_intervals(df):

    plt.figure(figsize=(12, 6))

    # Plot predicted values
    plt.plot(df['Date'], df['FORECAST'], label='Predicted', color='blue', marker='o')

    # Shade confidence intervals
    plt.fill_between(df['Date'], df['LOWER_95'], df['UPPER_95'], color='blue', alpha=0.2, label='95% CI')
    plt.fill_between(df['Date'], df['LOWER_97'], df['UPPER_97'], color='green', alpha=0.15, label='97% CI')
    plt.fill_between(df['Date'], df['LOWER_99'], df['UPPER_99'], color='red', alpha=0.1, label='99% CI')

    # Labels & legend
    plt.xlabel('Date')
    plt.ylabel('Predicted Value')
    plt.title('Forecast with Prediction Intervals')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()





