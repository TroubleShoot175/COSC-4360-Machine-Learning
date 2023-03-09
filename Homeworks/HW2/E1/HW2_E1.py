# Homework 2 - Exercise 1

# Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Main Code
df = pd.read_csv("avgHigh_jan_1895-2018.csv")
x = np.array(df["Date"])
y = np.array(df["Value"])
xLine = np.linspace(190000, 202000, 7)


xPred = [201901, 202301, 202401]

slope, yInt, r, p, stderr = stats.linregress(x, y)

yLine = (slope * xLine) + yInt

yPred = []
for xA in xPred:
     yPred.append((slope * xA) + yInt)

plt.scatter(x, y, color="b", label="Data Points")
plt.scatter(xPred, yPred, color="g", label="Predicted")
plt.plot(xLine, yLine, color="r", label="Model")

plt.xticks(np.linspace(190000, 202000, 7))
plt.title(f"January Average High Temperatures, Slope: {slope:.2f}, Intercept: {yInt:.2f}")
plt.legend(loc="lower right")

plt.show()
