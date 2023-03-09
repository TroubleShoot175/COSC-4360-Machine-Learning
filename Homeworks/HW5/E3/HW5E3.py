# Homework 5 - Exercise 3
# Libraries
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn.linear_model import RANSACRegressor

# Program
# Prediction Method
def myPredict(yInt, cA, xA):
    yPred = []
    for r in xA:
        pred = 0
        for i in range(len(cA)):
            pred = pred + cA[i] * r[i]
        pred = pred + yInt
        yPred.append(pred)
    return yPred

# Data Consolidation
df = pd.read_csv("materialsOutliers.csv")
#x = np.array(df.loc[:, "Time":"Temperature"])
x0 = np.array(df.loc[:, "Time"])
x1 = np.array(df.loc[:, "Pressure"])
x2 = np.array(df.loc[:, "Temperature"])
y = np.array(df.loc[:, "Strength"])

# Data Cleaning - RANSAC
rS0 = RANSACRegressor(residual_threshold=15, stop_probability=1.00).fit(x0.reshape(-1, 1), y)
x0 = np.delete(x0, np.logical_not(rS0.inlier_mask_), axis=0)

rS1 = RANSACRegressor(residual_threshold=15, stop_probability=1.00).fit(x1.reshape(-1, 1), y)
x1 = np.delete(x1, np.logical_not(rS1.inlier_mask_), axis=0)


rS2 = RANSACRegressor(residual_threshold=15, stop_probability=1.00).fit(x2.reshape(-1, 1), y)
x2 = np.delete(x2, np.logical_not(rS2.inlier_mask_), axis=0)

y = np.delete(y, np.logical_not(rS0.inlier_mask_), axis=0)
y = np.delete(y, np.logical_not(rS1.inlier_mask_), axis=0)
y = np.delete(y, np.logical_not(rS2.inlier_mask_), axis=0)

x = np.concatenate(x0, x1, x2, axis=1, dtype=np.float64)

# Multiple Linear Regression
lR = LinearRegression().fit(x, y)

