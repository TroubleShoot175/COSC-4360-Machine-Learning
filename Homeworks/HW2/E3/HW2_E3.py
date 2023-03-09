# Homework 2 - Exercise 3
# Libraries
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Main Code
df = pd.read_csv("wdbc.data.csv")
y = np.array(df.iloc[:, 1])
x = np.array(df.iloc[:, 2:32])

# Convert y to yI
yI = y
yI[yI=="M"] = 1
yI[yI=="B"] = 0

# Scale Data
x = StandardScaler().fit_transform(x)

# PCA 
n_components=2
pcaModel = PCA(n_components).fit(x)
x2D = pcaModel.transform(x)

# Plot
plt.scatter(x2D[:, 0], x2D[:, 1], c=yI)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"PCA = {n_components}, Variance: {pcaModel.explained_variance_}")
plt.show()
