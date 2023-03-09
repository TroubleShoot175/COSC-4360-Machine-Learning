# Lab 7 - Exercise 1
# Libraries
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

# Program
# Data Consolidation
df = pd.read_csv("Student-Pass-Fail.csv")
x = np.array(df.loc[:, "Self_Study_Daily":"Tution_Monthly"])
y = np.array(df.loc[:, "Pass_Or_Fail"])
xPred = np.array([[7, 28], [10, 34], [2, 39]])

# Logistic Regression
logReg = LogisticRegression().fit(x, y)

# Create Model
yPred = logReg.coef_ * xPred + logReg.intercept_

# Probability
prob = 1 / (1 + np.exp(-yPred))

# Print Out
print(f"Probabilites: \n{prob}")
