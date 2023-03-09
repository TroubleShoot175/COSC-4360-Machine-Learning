# Homework 3 - Exercise 2
# Libraries
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Data Conslidaton
df = pd.read_csv("hsbdemo.csv")

# Assign Labels to Array
y = np.array(df.loc[:, "prog"])

# Data Transformation
df.drop(["prog"], axis=1, inplace=True)
df.drop(["cid"], axis=1, inplace=True)
df.drop(["id"], axis=1, inplace=True)
df["gender"].replace(['male', 'female'], [0, 1], inplace=True)
df["ses"].replace(['low', 'middle', 'high'], [0, 1, 2], inplace=True)
df["schtyp"].replace(['public', 'private'], [0, 1], inplace=True)
df["honors"].replace(['enrolled', 'not enrolled'], [0, 1], inplace=True)

# Assign Data to Array
x = np.array(df.loc[:, "gender":"awards"])

# Data Reduction
n_components = 2

x = StandardScaler().fit_transform(x)
fitPcaModel = PCA().fit(x)
x2D = fitPcaModel.transform(x)

# Cumulative Sum and Variance Ratio
vR = fitPcaModel.explained_variance_ratio_
cS = np.cumsum(vR)

# Print Out
print(f"----- PCA Information -----")
print(f"Variance Ratio: {vR}")

# Plot - Cumulative Sum Of Variance Ratio
plt.plot([x+1 for x in range(10)], cS, color="#683d6e")
plt.show()



