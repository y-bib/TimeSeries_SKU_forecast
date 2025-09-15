import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import statsmodels.api as sm # for poisson regression
import statsmodels.formula.api as smf

from statsmodels.tsa.statespace.structural import UnobservedComponents # for state space method

import xgboost as xgb #for xboost

import os


def zscore(s, window, thresh=2.5, return_all=False):
    # If >80% zeros → clip 
    zero_ratio = (s == 0).mean()
    if zero_ratio > 0.8:
        q = s.quantile(0.98)
        s_clipped = s.clip(upper=q).astype(int)
        
        # Create the mask of unchanged values
        mask = (s <= q)
        
        if return_all:
            return s_clipped, mask
        else: return s_clipped
        
        
    # First pass: rolling median + MAD
    roll1 = s.rolling(window=window, min_periods=1, center=True)
    med1 = roll1.median()
    mad1 = roll1.apply(lambda x: (x - x.median()).abs().median(), raw=False)
    z1 = (s - med1) / (mad1 * 1.4826)
    #m1 = z1.between(-thresh, thresh)
    m1 = z1 < thresh
    s1 = s.where(m1, med1).astype(float)

    # # Second pass: recalc on cleaned data
    # roll2 = s1.rolling(window=window, min_periods=1, center=True)
    # med2 = roll2.median()
    # mad2 = roll2.apply(lambda x: (x - x.median()).abs().median(), raw=False)
    # z2 = (s1 - med2) / (mad2 * 1.4826)
    # m2 = z2.between(-thresh, thresh)
    # s2 = s1.where(m2, med2)

    # mask = m1 & m2
    # s2 = s.where(mask, med2)

    if return_all:
        return med1, m1
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

