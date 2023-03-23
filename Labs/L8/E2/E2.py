# Lab 8 - Exercise 2
# Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os
import sys

np.set_printoptions(threshold=sys.maxsize)

# Program
def getClasses(labelsDic: dict, listValsToLabel: list)-> list:
    labeledValues = []
    for val in listValsToLabel:
        labeledValues.append(labelsDic[val])
    return labeledValues

# Training Data Consolidation
dfTrain = pd.read_csv("fashion-mnist_train.csv")
xTrain = np.array(dfTrain.loc[:, "pixel1":"pixel784"])
yTrain = np.array(dfTrain.iloc[:, 0])

# Train Logistic Regression Model
logReg = LogisticRegression(multi_class="multinomial").fit(xTrain, yTrain)

# Prepare Files In ToClassify
path = os.getcwd() + "\ToClassify"
toClassifyFiles = list(path + "\\" + fileTitle for fileTitle in os.listdir(path))
xTest = []

for file in toClassifyFiles:
    if os.path.splitext(file)[1] == ".bmp":
        xTest.append(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY).reshape(1, 28 * 28))
    else:
        xTest.append(cv2.resize(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY), (28, 28)).reshape(1, 28 * 28))

xTest = np.array(list(arr.reshape(-1) for arr in xTest))

# Classification
yPred = logReg.predict(xTest)
yPredProbs = logReg.predict_proba(xTest)

# Print Out
labelDic = {0:"T-shirt/top", 1:"Trouser", 2:"Pullover", 3:"Dress", 4:"Coat", 5:"Sandal", 6:"Shirt", 7:"Sneaker", 8:"Bag", 9:"Ankle boot"}
print(f"Classifications:")
for item in zip(list(fileTitle for fileTitle in os.listdir(path)), getClasses(labelDic, yPred), yPredProbs):
    print(f"File: {item[0]} | Classified As: {item[1]} | With Probability: {'{0:.2f}'.format(max(item[2]))}")
