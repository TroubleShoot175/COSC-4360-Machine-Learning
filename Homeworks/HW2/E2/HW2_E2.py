# Homework 2 - Exercise 2

# Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error

# Custom Built train_test_split()
def myTestTrainSplit(x, y, sP):
    aTD = round(len(x) * sP)
    xTest = x[-aTD:]
    yTest = y[-aTD:]
    xTrain = x[:-aTD]
    yTrain = y[:-aTD]
    return xTrain, xTest, yTrain, yTest

# Main Code
df = pd.read_csv("avgHigh_jan_1895-2018.csv")
df.drop(["Anomaly"], axis=1, inplace=True)
x = np.array(df["Date"])
y = np.array(df["Value"])

# Ask user for test size.
userIn = float(input("Enter test size: "))

# Split training data
xTrain, xTest, yTrain, yTest = myTestTrainSplit(x, y, userIn)

# Create Regression from x and y training data
slope, yInt, r, p, stderr = stats.linregress(xTrain, yTrain)

# To Create Model Line
xLine = np.linspace(190000, 200000, 7)
yLine = (slope * xLine) + yInt

# To Predict Temperatures
yPred = []
for xT in xTest:
    yPred.append((slope * xT) + yInt)

# ----------- Print Out -----------

print(f"\nSlope: {slope}\nY-Int: {yInt}\nCorrelation: {r}\nP-Value: {p}\nStandard Error: {stderr}\n")

for i in range(0, 19):
    print(f"Actual: {yTest[i]} Predicted: {yPred[i]}")

RMSE = (mean_squared_error(yTest, yPred))**0.5

print(f"Root Mean Square Error: {(mean_squared_error(yTest, yPred))**0.5}")

# ---------- Print Out End ----------

# Plot Data
plt.scatter(xTrain, yTrain, color="b", label="Train")
plt.scatter(xTest, yTest, color="g", label="Test")
plt.plot(xLine, yLine, color="r", label="Model")
plt.xticks(np.linspace(190000, 202000, 7))
plt.title(f"Slope: {slope:.2f}, Intercept: {yInt:.2f}, Test Size: {userIn}, RMSE: {RMSE}")
plt.legend(loc="lower right")
plt.show()
