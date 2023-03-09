# Lab 5 - Exercise 3
# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from scipy import stats
import seaborn as sns

# Data Consolidation
df = fetch_california_housing(as_frame=True).frame

# Data Visualiaztion
sns.pairplot(df.loc[:, "MedInc":"AveOccup"])
plt.show()
