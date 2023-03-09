# Lab 7 - Exercise 2
# Libraries
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

# Program
# Custom Built train_test_split()
def myTestTrainSplit(x, y, sP):
    aTD = round(len(x) * sP)
    xTest = x[-aTD:]
    yTest = y[-aTD:]
    xTrain = x[:-aTD]
    yTrain = y[:-aTD]
    return xTrain, xTest, yTrain, yTest

# myConfusionMatrix
def myConfusionMatrix(a, p):
    mF = np.zeros([len(np.unique(a)), len(np.unique(a)) + 1])

    cnt = 0
    for i in range(len(mF)):
        mF[i][0] = cnt
        cnt = cnt + 1

    for i in range(len(mF)):
        for j in range(len(a)):
            if a[j] == mF[i][0]:
                if a[j] == p[j]:
                    mF[i][a[j] + 1] = mF[i][a[j] + 1] + 1
                else:
                    mF[i][p[j] + 1] = mF[i][p[j] + 1] + 1

    cM = mF[:, -len(np.unique(a)):]
    return cM


# myTestTrainSplit
def myTestTrainSplit(x, y, sP):
    aTD = round(len(x) * sP)
    xTest = x[-aTD:]
    yTest = y[-aTD:]
    xTrain = x[:-aTD]
    yTrain = y[:-aTD]
    return xTrain, xTest, yTrain, yTest

# Data Consolidation
df = pd.read_csv("Student-Pass-Fail.csv")
x = np.array(df.loc[:, "Self_Study_Daily":"Tution_Monthly"])
y = np.array(df.loc[:, "Pass_Or_Fail"])

# Data Split
xTrain, xTest, yTrain, yTest = myTestTrainSplit(x, y, 0.25)

# Logistic Regression
logReg = LogisticRegression().fit(xTrain, yTrain)

# Create Model
yPred = logReg.predict(xTest)

# Create Confusion Matrix
print(myConfusionMatrix(yTest, yPred))
