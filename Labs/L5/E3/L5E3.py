# Lab 5 - Exercise 3
# Libraries
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from scipy import stats

# Data Consolidation
df = fetch_california_housing(as_frame=True).frame
y = np.array(df.loc[:, "MedHouseVal"])
x = np.array(df.loc[:, "MedInc":"Longitude"])

# Regression
max = -100
for i in range(len(x[0])):
    s, yInt, r, p, stdErr = stats.linregress(df.iloc[:, i], y)
    print(f"r: {r} | {df.columns[i]}")
    if r > max:
        max = r
        mI = i

# Print Out
print(f"r: {max} | {df.columns[mI]}")