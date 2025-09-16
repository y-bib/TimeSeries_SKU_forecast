import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import xgboost as xgb #for xboost
from sklearn.model_selection import TimeSeriesSplit
from errors_custom import errors

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from errors_custom import errors

def xgb_cv(df_cl, max_lag=7, test_size=30, plot=True):
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

    X = df_tmp.drop(columns=['y'])
    y = df_tmp['y'].values

    # -------------------------
    # 2) Time series cross-validation
    # -------------------------
    tscv = TimeSeriesSplit(test_size=test_size)
    
    y_preds = np.full_like(y, fill_value=np.nan, dtype=float)
    all_index = []  # collect all test indices

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

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

        pred_mean = model.predict(X_test)
        y_preds[test_index] = pred_mean
        all_index.extend(test_index)  # store indices

    all_index = np.array(all_index)  # convert to numpy array

    # -------------------------
    # 3) Calculate errors
    # -------------------------
    err = errors(y[all_index], y_preds[all_index], np.round(y_preds[all_index]).astype(int))

    results = {
        "pred_mean": y_preds,
        "pred_integer": np.round(y_preds).astype(int),
        "mae_mean": err["mae_mean"],
        "mae_integer": err["mae_integer"],
        "mape_mean": err["mape_mean"],
        "mape_integer": err["mape_integer"],
        "var_test_sample": err["var_test_sample"]
    }

    # -------------------------
    # 4) Plot observed vs predicted
    # -------------------------
    if plot:
        plt.figure(figsize=(12, 5))
        
        # Plot the entire observed series
        plt.plot(df_cl['VALUE2'].values, label="Observed (all data)", color='blue', alpha=0.6)
        
        # Plot predicted values (only where predictions exist)
        plt.plot(y_preds, label="Predicted mean", color='red', marker='x', linestyle='None')
        plt.plot(np.round(y_preds), label="Predicted integer", color='green', marker='s', linestyle='None', alpha=0.7)
        
        plt.xlabel("Time index")
        plt.ylabel("Counts")
        plt.title("Observed vs Predicted Values (XGBoost Poisson Forecast)")
        plt.legend()
        plt.show()
    return results
