# Homework 7 - Exercise 1
# Libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Program
# Data Consolidation
df = pd.read_csv('vehicles.csv')

# Data Cleaning
df.drop(['make'], axis=1, inplace=True)

# Data Assignment
y = np.array(df.loc[:, "mpg"])
x = np.array(df.loc[:, "cyl":"carb"])

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

# Retireve 5 Most Important Features
fL = list(map(lambda x : abs(x), cCS))
fLC = list(map(lambda x : abs(x), cCS))
fLC.sort()
mIFVL = fLC[-5:]
mIFIL = []
for i in range(len(mIFVL)):
    mIFIL.append(fL.index(mIFVL[i]))

mIFL = []
colNames = list(df.columns)
colNames.pop(0)
for i in range(len(mIFIL)):
    mIFL.append(colNames[mIFIL[i]])

# Print Out
print(mIFL)
