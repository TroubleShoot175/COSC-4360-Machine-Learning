# Homework 3 - Exercise 3
# Libraries
from sklearn.linear_model import RANSACRegressor
from scipy.stats import linregress
import numpy as np
import pandas as pd

x = np.array([100, 150, 185, 235, 310, 370, 420, 430, 440, 530, 600, 634, 718, 750, 850, 903, 978, 1010, 1050, 1990])
y = np.array([12300, 18150, 20100, 23500, 31005, 359000, 44359, 52000, 53853, 61328, 68000, 72300, 77000, 89379, 93200, 97150, 102750, 115358, 119330, 323989])

s, yInt, r, p, stdErr = linregress(x, y)

reg = RANSACRegressor(random_state=0).fit(x, y)

params = reg.get_params()
