# Lab 5 - Exercise 1
# Libraries
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

# Data Consolidation
df = fetch_california_housing(as_frame=True).frame
y = np.array(df.loc[:, "MedHouseVal"])
x = np.array(df.loc[:, "MedInc":"Longitude"])

# Regression
reg = LinearRegression().fit(x, y)

# Prediction
xPred = np.array([[8.3153, 41.0, 6.894423, 1.053714, 323.0, 2.533576, 37.88, -122.23]])
yPred = reg.predict(xPred)

# Data Visualization
print(f"Linear Regression:")
print(f"Coefficent: \n{reg.coef_}")
print(f"y-intercept: {reg.intercept_}")
print(f"Prediction: {yPred}")

