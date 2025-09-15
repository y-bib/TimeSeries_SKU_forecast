import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import statsmodels.api as sm # for poisson regression
import statsmodels.formula.api as smf

from statsmodels.tsa.statespace.structural import UnobservedComponents # for state space method

import xgboost as xgb #for xboost

import os


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
