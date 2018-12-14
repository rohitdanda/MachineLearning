import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
##Decision are not so accurate in REgression so this is just practise

datasets = pd.read_csv("DecisionTreeRegression/Position_Salaries.csv")

X_dependent = datasets.iloc[:,1:2].values
Y_independent = datasets.iloc[:,2].values

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_dependent,Y_independent)

ypredic = regressor.predict(6.5)

# plot.scatter(X_dependent,Y_independent)
# plot.plot(X_dependent,regressor.predict(X_dependent))
# plot.title("Decsion Tree Regression")
# plot.xlabel("Salary")
# plot.ylabel("Expereiece")
# plot.show()

##For the Higher Resolution of Plotting

X_grid = np.arange(min(X_dependent),max(X_dependent),0.01)
X_grid=X_grid.reshape((len(X_grid),1))

plot.scatter(X_dependent,Y_independent)
plot.plot(X_grid,regressor.predict(X_grid))
plot.title("Decision Tree Regression")
plot.xlabel("Salary")
plot.ylabel("Expereicne")
plot.show()