import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import math as mt
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm
from sklearn.metrics import confusion_matrix

datasets = pd.read_csv("Example/data.csv")

x = datasets.iloc[:,0:2].values
y = datasets.iloc[:,2].values

regression = LinearRegression()
regression.fit(x,y)

ypred = regression.predict([[3,8]])

print(ypred)
# cm = confusion_matrix(y,ypred)

print(mt.ceil(ypred))