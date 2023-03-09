# Lab 3 - Exercise 1
# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Code
# Names for Columns wihtin dataframe
names =  ["class", "Alcohol","Malic Acid","Ash","Acadlinity","Magnisium","Total Phenols","Flavanoids",
"NonFlavanoid Phenols", "Proanthocyanins", "Color Intensity", "Hue", "OD280/OD315", "Proline"]

# Read data from "wine.csv" and assign to df
df = pd.read_csv("wine.csv", header=None, names=names)

# Split the data in the df to two arrays x and y. x being for the wine attributes and y for the data points classificaiton
x = np.array(df.loc[:, "Malic Acid":"Proline"])
y = np.array(df["class"])

# Seperate the data into the training and test data sets
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.20)

# List to store scores of classificaions and assign the range generator to the value kR
scores = []
kR = range(1, 10)

# Iterate through the range of k values to generate the scores for the models then append them to the "scores" list
for k in kR:
    # Classify the data using K-Nearest Neighbors (KNN) Algorithm
    # Create the KNN Classifier Model
    knnModel = KNeighborsClassifier(n_neighbors=k)
    # Fit the KNN model to the training data
    knnModel.fit(xTrain, yTrain)
    # Predict the classification of the test data
    yPred = knnModel.predict(xTest)
    # Test the accuracy of model by taking the predicted classifications and comparing them to the actual classifications then append the score to the scores array
    scores.append(accuracy_score(yTest, yPred))

# Plot the ranges and the scores on to a line plot
plt.plot(kR, scores)
plt.title("")
plt.xlabel("Value of K for KNN Model")
plt.ylabel("Test Accuracy")
plt.show()