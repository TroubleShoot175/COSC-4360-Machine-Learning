# Lab Two - Exercise One

# Libraries
import matplotlib as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

# Code
names = ["class", "Alcohol","Malic Acid","Ash","Acadlinity","Magnisium","Total Phenols","Flavanoids", "NonFlavanoid Phenols", "Proanthocyanins", "Color Intensity", "Hue", "OD280/OD315", "Proline" ]

newWines = np.array([[14.23,1.71,2.43,15.6,127,2.8,3.06,.28,2.29,5.64,1.04,3.92,1065],
                     [12.64,1.36,2.02,16.8,100,2.02,1.41,.53,.62,5.75,.98,1.59,450],
                     [12.53,5.51,2.64,25,96,1.79,.6,.63,1.1,5,.82,1.69,515],
                     [13.49,3.59,2.19,19.5,88,1.62,.48,.58,.88,5.7,.81,1.82,580]])

df = pd.read_csv("wine.csv", header=None, names=names)
x = np.array(df.iloc[:, 1:13])
y = np.array(df["class"])

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
pred = knn.predict(x_test)

print(f"\nModel Accuracy Score: {accuracy_score(y_test, pred)}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, pred)}")

print(f"\nPrediction of the classes for each of the four wines: {knn.predict(newWines[:, 1:13])}")
