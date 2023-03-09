# Lab 4 - Exercise 1
# Libraries
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import seaborn as sns

# Main Code
# Load Data
df = pd.read_csv("recipes_muffins_cupcakes_scones.csv")
y = np.array(df["Type"])
x = np.array(df.iloc[:, 1:9])

# PCA Components = 2
x = StandardScaler().fit_transform(x)
fitPcaModel = PCA(n_components=2).fit(x)
x2D = fitPcaModel.transform(x)

# PCA Components = 4
n_components=4
fitPcaModel2 = PCA(n_components).fit(x)
x4D = fitPcaModel2.transform(x)

# Cumulative Sum and Variance Ratio
vR = fitPcaModel2.explained_variance_ratio_
cS = np.cumsum(vR)

# Convert y to yI
yI = y
yI[yI=="Muffin"] = 2
yI[yI=="Scone"] = 1
yI[yI=="Cupcake"] = 0

# Assign data to PCX
pc1 = x2D[:, 0]
pc2 = x2D[:, 1]

# PC Heatmap Data
df = pd.DataFrame(fitPcaModel.components_)
xAxisLabels = ["Flour", "Milk", "Sugar", "Butter", "Egg", "Baking Powder", "Vanilla", "Salt"]
yAxisLabels = ["PC1", "PC2"]

# Find Features with Highest and Lowest Variation
pc1F = np.array(df.abs().iloc[0, :])
pc2F = np.array(df.abs().iloc[1, :])

pc1Dic = dict(zip(xAxisLabels, pc1F))
pc2Dic = dict(zip(xAxisLabels, pc2F))

pc1Max = max(pc1Dic, key=pc1Dic.get)
pc1Min = min(pc1Dic, key=pc1Dic.get)
pc2Max = max(pc2Dic, key=pc2Dic.get)
pc2Min = min(pc2Dic, key=pc2Dic.get)


# PC Covarience Matrix 
covMat = np.cov(x.T)

# Print Out
print(f"----- PCA Information -----")
print(f"Variance Ratio: {fitPcaModel.explained_variance_ratio_}")
print(f"Cumulative Sum: {np.cumsum(fitPcaModel.explained_variance_ratio_)}\n")
print(f"----- PC1 -----\nHighest Variation Feature: {pc1Max}\nLowest Variation Feature: {pc1Min}\n")
print(f"----- PC2 -----\nHighest Variation Feature: {pc2Max}\nLowest Variation Feature: {pc2Min}")

# PC Subplots
fig, ax = plt.subplots(2, 2, figsize=(12, 10))
p1, p2, p3, p4 = ax.flatten()

# P1 - PCs Scatter Plot
p1.scatter(pc1, pc2, c=yI, cmap=sns.cubehelix_palette(as_cmap=True))

# P2 - PC Largest Variation Heatmap Plot
sns.heatmap(df, cmap=sns.cubehelix_palette(as_cmap=True), xticklabels=xAxisLabels, yticklabels=yAxisLabels, ax=p2)

# P3 - PC Covariance Heatmap
sns.heatmap(covMat, cmap=sns.cubehelix_palette(as_cmap=True), xticklabels=xAxisLabels, yticklabels=xAxisLabels, ax=p3)

# P4 - Cumulative Sum Of Variance Ratio
p4.plot([x+1 for x in range(n_components)], cS, color="#683d6e")

# Show And Design Plots
fig.suptitle("PCA Plots")
p1.title.set_text("PC1 & PC2")
p1.set(xlabel="PC1", ylabel="PC2")
p2.title.set_text("Highest Variation")
p3.title.set_text("PCs Covariance Map")
p4.title.set_text("Cumulative Sum")
p4.set(xlabel="PC1", ylabel="PC2")
plt.show()

