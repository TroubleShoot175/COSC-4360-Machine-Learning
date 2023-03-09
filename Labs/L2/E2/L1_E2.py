# Lab One - Exercise Two

# Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Code
names = ["sepal_length", "sepal_width", "petal_length", "pedal_width", "class"]
df = pd.read_csv("iris.data.csv", header=None, names=names)
x = np.array(df.iloc[:, 0])
y = np.array(df.iloc[:, 1])

df = df.drop("class", axis=1)
pca = PCA(n_components=2).fit(df)
data_2d = pca.transform(df)

kmeans = KMeans(n_clusters=3).fit(data_2d)
centroids = kmeans.cluster_centers_

for i in range(0, data_2d.shape[0]):
    if kmeans.labels_[i] == 0:
        plt.scatter(data_2d[i, 0], data_2d[i, 1], c='green')
    elif kmeans.labels_[i] == 1:
        plt.scatter(data_2d[i, 0], data_2d[i, 1], c='yellow')
    elif kmeans.labels_[i] == 2:
        plt.scatter(data_2d[i, 0], data_2d[i, 1], c='purple')

plt.show()