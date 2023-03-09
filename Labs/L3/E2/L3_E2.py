# Lab 3 - Exercise 2

# Libraries
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# import seaborn as sns
# import matplotlib.pyplot as plt

# Code
df = pd.read_csv("breast-cancer-wisconsin.data.csv")

# Replace "?" with NaN
df.replace(to_replace="?", value=np.nan, inplace=True)

# Drop rows with a NaN in them
df.dropna(axis=0, how="any", inplace=True)

# Assign the columns to x and y, the former being features and the latter being classificaion
x = np.array(df.iloc[:, 1:9])
y = np.array(df.iloc[:, 10])

# Split the data into training data and testing data
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.30)

# Create the KNN Model to be trained on the training data
knnModel = KNeighborsClassifier(n_neighbors=5)

# Fit the data to the created KNN Model
knnModel.fit(xTrain, yTrain)

# Predict the classification of the test data
yPred = knnModel.predict(xTest)

# Create Confusion Matrix and assign it to the value cM to ploted as a heatmap
cM = confusion_matrix(yTest, yPred)

# Print Confusion Matrix
print(f"Confusion Matrix:\n{confusion_matrix(yTest, yPred)}")

# ----- This section is for a Heat Map Visualization of the Confusion Matrix -----

# Create Labels for Heatmap
# cMNames = ['True Neg','False Pos','False Neg','True Pos']
# cMCnt = ["{0:0.0f}".format(value) for value in cM.flatten()]
# cMPercentages = ["{0:.2%}".format(value) for value in cM.flatten()/np.sum(cM)]
# labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(cMNames,cMCnt,cMPercentages)]

# Reshape convert list to numpy array then reshape to a 2 by 2 array
# labels = np.asarray(labels).reshape(2,2)

# Create the Heatmap for the Confusion Matrix to see the accuracy of KNN Model
# sns.heatmap(cM/np.sum(cM), annot=labels, fmt="", cmap="Blues")

# Label Heatmap
# plt.title("Confusion Matrix")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")

# Show Heatmap
# plt.show()
