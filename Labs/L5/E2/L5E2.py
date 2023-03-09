# Lab 5 - Exercise 2
# Libraries
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Data Consolidation
df = fetch_california_housing(as_frame=True).frame
y = np.array(df.loc[:, "MedHouseVal"])
x = np.array(df.loc[:, "MedInc":"Longitude"])

# Regression
reg = LinearRegression().fit(x, y)

#b = []
#for i in range(len(reg.coef_)):
#    b.append(reg.coef_[i])

yInt = reg.intercept_
b1 = reg.coef_[0]
b2 = reg.coef_[1]

x1 = np.array(df.loc[:, "MedInc"])
x2 = np.array(df.loc[:, "HouseAge"])

X1, X2 = np.meshgrid(x1, x2)

def f(X1, X2, yInt, b1, b2):
    return (yInt + b1*X1 + b2*X2)

Z = f(X1, X2, yInt, b1, b2)

# Data Visualization
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot_wireframe(X1, X2, Z, color = 'green')
ax.set_title('3D Graph')
plt.show()
