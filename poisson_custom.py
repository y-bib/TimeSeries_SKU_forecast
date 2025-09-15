#from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

import statsmodels.api as sm # for poisson regression
import statsmodels.formula.api as smf
from statsmodels.tsa.statespace.structural import UnobservedComponents # for state space method
import xgboost as xgb #for xboost
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error


from outliners_fix import zscore, plot_zscore, cap_outliers, plot_cap_outliers
from data_split import get_season, date_split
from errors_custom import errors
from state_space_custom import state_space, fit_ucm_and_evaluate, state_space_out
from xgb_poisson_custom import fit_xgb_poisson2, fit_xgb_poisson2_out
from corr_custom import means_and_cor





# returns data frame for one SKU only
def sku_extract(df, SKU = '991234-A'):
    df_cl = df[['DATE','VALUE','ITEM_CODE']][df.ITEM_CODE == SKU].copy()
    df_cl.reset_index(drop=True, inplace=True)
    return df_cl

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


def process_sku(df, SKU):
    print(f"Processing SKU: {SKU}")

    # 1) Extract SKU data
    df_cl = sku_extract(df, SKU)

    # 2) Compute z-score for VALUE
    n=max(round(len(df_cl.index)/10),30)
    df_cl['VALUE2'] = zscore(df_cl['VALUE'], window=n)

    avg, m = zscore(df_cl['VALUE'], window=n, return_all=True)
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
    max_run = df_cl["VALUE2"].groupby((df_cl["VALUE2"] != df_cl["VALUE2"].shift()).cumsum()).size().max()

    results = fit_xgb_poisson2(df_cl, test_size=0.2, max_lag=max_run+1, plot=True)  # other good option space_state function

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
    
    df_pred = fit_xgb_poisson2_out(df_cl, df_pred, max_lag=max_run+1, n=30)

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





