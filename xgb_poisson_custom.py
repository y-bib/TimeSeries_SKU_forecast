import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# import statsmodels.api as sm # for poisson regression
# import statsmodels.formula.api as smf

import xgboost as xgb #for xboost

from errors_custom import errors


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


#Xboost with all seasonality and lags
def fit_xgb_poisson2(df_cl, test_size=0.2, max_lag=7, plot=True):
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
# possible call
# results = fit_xgb_poisson2(df_cl, test_size=0.2, max_lag=3)
# print(f"MAE (mean): {results['mae_mean']:.3f}")
# print(f"MAE (integer): {results['mae_integer']:.3f}")
# print(f"MAPE (%) mean: {results['mape_mean']:.2f}%")
# print(f"MAPE (%) integer: {results['mape_integer']:.2f}%")


def fit_xgb_poisson2_out(df_cl, df_pred, max_lag=7, n=30):

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
