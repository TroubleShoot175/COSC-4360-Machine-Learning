# Homework 3 - Exercise 1
# Libraries
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import numpy as np

# Data Conslidaton
df = pd.read_csv("hsbdemo.csv")

# Assign Labels to Array
y = np.array(df.loc[:, "prog"])

# Data Cleaning
df.drop(["prog"], axis=1, inplace=True)
df.drop(["cid"], axis=1, inplace=True)
df.drop(["id"], axis=1, inplace=True)

# Data Transformation
df["gender"].replace(['male', 'female'], [0, 1], inplace=True)
df["ses"].replace(['low', 'middle', 'high'], [0, 1, 2], inplace=True)
df["schtyp"].replace(['public', 'private'], [0, 1], inplace=True)
df["honors"].replace(['enrolled', 'not enrolled'], [0, 1], inplace=True)

# Assign Data to Array
x = np.array(df.loc[:, "gender":"awards"])

# Split Data into Testing and Training
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.10)

# Model Training
trainedKnnModel = KNeighborsClassifier().fit(xTrain, yTrain)

# Preditction
yPred = trainedKnnModel.predict(xTest)

# Print Out
print(f"Model Accuracy Score: {accuracy_score(yTest, yPred)}")
print(f"Confusion Matrix: \n{confusion_matrix(yTest, yPred)}")

