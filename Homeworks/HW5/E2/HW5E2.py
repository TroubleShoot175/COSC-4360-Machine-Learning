# Homework 5 - Exercise 2
# Libraries
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Program
# Data Consolidation
df = pd.read_csv("materials.csv")
x = np.array(df.loc[:, "Pressure":"Temperature"])
pressure = np.array(df.loc[:, "Pressure"])
temperature = np.array(df.loc[:, "Temperature"])

y = np.array(df.loc[:, "Strength"])

# Linear Regression
reg = LinearRegression().fit(x, y)

# 3D Mesh-grid Plot
x1, x2 = np.meshgrid(pressure, temperature)
z = reg.intercept_ + reg.coef_[0] * x1 + reg.coef_[1] * x2
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot_wireframe(x1, x2, z, color = 'blue')

#3D Scatter Plot
ax.scatter3D(pressure, temperature, y, c=y, cmap='Greens')
ax.set_title('3D Graph')
ax.set_xlabel("Pressure")
ax.set_ylabel("Temperature")
ax.set_zlabel("Strength")
plt.show()
