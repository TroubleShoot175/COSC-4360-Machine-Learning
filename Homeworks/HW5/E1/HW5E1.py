# Homework 5 - Exercise 1

# Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Program
# Data Consolidation
df = pd.read_csv("materials.csv")
y = np.array(df.loc[:, "Strength"])
x = np.array(df.loc[:, "Time":"Temperature"])
xPred = np.array([[32.1, 37.5, 128.95], [36.9, 35.37, 130.03]])

# Standardization
SX = StandardScaler().fit_transform(x, y)

# Linear Regression
lRA = LinearRegression().fit(SX, y)
lR = LinearRegression().fit(x, y)

# Correlation Coefficients
# Non-Standardized Data
cC = []
for r in lR.coef_:
    cC.append(r)

# Standardized Data
cCS = []
for r in lRA.coef_:
    cCS.append(r)

# Retireve Most Important Feature
fL = list(map(lambda x : abs(x), cCS))
fL.sort()
mIFV1 = fL[-1]
mIFV2 = fL[-2]
mIFI1 = fL.index(mIFV1)
mIFI2 = fL.index(mIFV2)
mIF1 = list(df.columns)[mIFI1 + 1]
mIF2 = list(df.columns)[mIFI2 + 1]

# y-intercept
yInt = lR.intercept_

# Prediction
yPred = []
for row in xPred:
    #print(f"yInt: {yInt}, V1: {row[0]}, C1: {cC[0]}, V2: {row[1]}, C2: {cC[1]}, V3: {row[2]}, C3: {cC[2]}")
    yPred.append(yInt + (cC[0] * row[0]) + (cC[1] * row[1]) + (cC[2] * row[2]))

# Print Out
print(f"Correlation Coefficients: \n{cC}\n\nyInt: \n{yInt}\n\nxPred: \n{xPred}\n\nyPred: \n{yPred}\n\nMost Important Features:\n{mIF1, mIF2}")
