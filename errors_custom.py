import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import statsmodels.api as sm # for poisson regression
import statsmodels.formula.api as smf

from statsmodels.tsa.statespace.structural import UnobservedComponents # for state space method
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
import os



# from sklearn.metrics import mean_absolute_error
# from statsmodels.tsa.statespace.structural import UnobservedComponents
# from sklearn.metrics import mean_absolute_percentage_error


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
