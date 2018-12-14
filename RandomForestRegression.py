import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from sklearn.ensemble import RandomForestRegressor

datasets = pd.read_csv("RandomForestTreeRegression/Position_Salaries.csv")

X_dependent = datasets.iloc[:,1:2].values
Y_independent = datasets.iloc[:,2].values

regressor = RandomForestRegressor(n_estimators=300,random_state=0)
regressor.fit(X_dependent,Y_independent)
regressor.predict(6.5)

x_grid = np.arange(min(X_dependent),max(X_dependent),0.1)
x_grid = x_grid.reshape((len(x_grid),1))

plot.scatter(X_dependent,Y_independent, color='red')
plot.plot(x_grid,regressor.predict(x_grid))
plot.title("Random Forest Regression")
plot.xlabel("Salary")
plot.ylabel("Expereience")
plot.show()