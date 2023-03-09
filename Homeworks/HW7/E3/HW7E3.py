# Homework 7  - Exercise 3
# Libraries
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

# Program
xPred = np.array([6, 163, 111, 3.9, 2.77, 16.45, 0, 1, 4, 4])

# Data Consolidation
df = pd.read_csv("vehicles.csv")

# Data Cleaning
df.drop(["make"], axis=1, inplace=True)

# Data Assignment
x = np.array(df.loc[:, "cyl":"carb"])
y = np.array(df.loc[:, "mpg"])

# Linear Regression
reg = LinearRegression().fit(x, y)

# Prediction 
yPred = reg.predict(xPred.reshape(1, -1))

# Print Out
print(f"Prediction: {yPred}")
