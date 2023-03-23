# Lab 8 - Exercise 1
# Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Program
# Data Consolidation
dfTrain = pd.read_csv("fashion-mnist_train.csv")
dfTest = pd.read_csv("fashion-mnist_test.csv")
xTrain = np.array(dfTrain.loc[:, "pixel1":"pixel784"])
yTrain = np.array(dfTrain.iloc[:, 0])
xTest = np.array(dfTest.loc[:, "pixel1":"pixel784"])
yTest = np.array(dfTest.iloc[:, 0])

# Logistic Regression
logReg = LogisticRegression().fit(xTrain, yTrain)

# Prediction
yPred = logReg.predict(xTest)

# Model Metrics
aS = accuracy_score(yTest, yPred)
cM = confusion_matrix(yTest, yPred)

# Print Out
print(f"Accuracy Score: \n{aS}")
print(f"Confusion Matrix:\n{cM}")

# Confusion Matrix Visualized
cMDisplay = ConfusionMatrixDisplay(confusion_matrix=cM, display_labels=logReg.classes_)
cMDisplay.plot()
plt.title("Logistic Regression - Confusion Matrix")
plt.show()
