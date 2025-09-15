
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import statsmodels.api as sm # for poisson regression
import statsmodels.formula.api as smf

from statsmodels.tsa.statespace.structural import UnobservedComponents # for state space method

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
