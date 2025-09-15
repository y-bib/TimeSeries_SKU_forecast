import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


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
