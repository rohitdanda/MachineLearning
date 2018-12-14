import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

datasets = pd.read_csv('PolynomialRegression/Position_Salaries.csv')

# Divide the data in to X and Y

X_dependent = datasets.iloc[:,1:2].values
Y_indenpendent = datasets.iloc[:,2].values # this value should always have second index without : because it represnet vector

# Linear Regression model to compare with Polynomial Regression
linearRegressor = LinearRegression()
linearRegressor.fit(X_dependent,Y_indenpendent)
Y_predict = linearRegressor.predict(X_dependent)

#Adding the polynomial Column to dataset
polynomial = PolynomialFeatures(degree=4)
X_dependent_ploy = polynomial.fit_transform(X_dependent)
polynomial.fit(X_dependent_ploy,Y_indenpendent)
linearRegression1 = LinearRegression()
linearRegression1.fit(X_dependent_ploy,Y_indenpendent)

#Plotting the Graph to see exact Match
plot.scatter(X_dependent,Y_indenpendent,color='red')
plot.plot(X_dependent,linearRegression1.predict(polynomial.fit_transform(X_dependent)))
plot.title("Simple Linear")
plot.xlabel("Salary")
plot.ylabel("Years")

# preditct the particular value in the Linear Regression
linearRegressor.predict(6.5)

# predict the particular value in the Polynomial Regression
linearRegression1.predict(polynomial.fit_transform(6.5))