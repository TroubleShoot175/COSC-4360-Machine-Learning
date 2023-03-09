# Lab 6 - Exercise 1
# Libraries
import pandas as pd
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)

# Program
# Data Consolidation
df = pd.read_csv("auto-mpg.csv")

# Data Cleaning
df.drop("origin", axis=1, inplace=True)
df.drop("car name", axis=1, inplace=True)
df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)

# Data Assignment
x = np.array(df.loc[:, "cylinders":"model year"]).astype(np.float64)
y = np.array(df.loc[:, "mpg"]).astype(np.float64)

# Regression Using Matrix Algebra To Find Coefficients
x = np.insert(x, 0, list(1 for _ in range(len(x))), axis=1)
xTX = np.matmul(x, x.T)
A = np.linalg.inv(xTX)
B = np.matmul(x.T, y)
b = np.matmul(A, B)



