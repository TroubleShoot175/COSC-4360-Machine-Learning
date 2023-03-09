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

