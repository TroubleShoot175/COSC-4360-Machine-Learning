# Lab 7 - Exercise 3
# Libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Program
# Data Consolidation
df = pd.read_csv('E3\Bank-data.csv')
y = np.array(df.loc[:, "y"])
x = np.array(df.loc[:, "interest_rate":"duration"])
xPred = np.array([[1.335, 0, 1, 0, 0, 109], [1.25, 0, 0, 1, 0, 279]])

# Logistic Regression
logReg = LogisticRegression().fit(x, y)

# Create Model
yModel = logReg.coef_ * xPred + logReg.intercept_

# Calculate Odds
logOdds = np.exp(logReg.coef_)

# Calculate Probability
logProb = np.exp(yModel) / (np.exp(yModel) + 1)

# Prediction
yPred = logReg.predict(xPred)

# Print Out
print(f"Probabilities: \n{logProb}")
print(f"Prediction: \n{yPred}")
