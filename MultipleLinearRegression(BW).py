import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm

dataSets = pd.read_csv('Multiple_LinearRegression/50_Startups.csv')

X_dependent = dataSets.iloc[:,0:4].values
y_independent = dataSets.iloc[:,4].values

# Encode the Label as state are not in our Format
labelencoderx = LabelEncoder()
X_dependent[:,3]=labelencoderx.fit_transform(X_dependent[:,3])

# Now create an Dummy values from the Label Encoder
oneHotEncoderX = OneHotEncoder(categorical_features=[3])
X_dependent=oneHotEncoderX.fit_transform(X_dependent).toarray()

# Avoid the Dummy Trap by removing one index linear regression take care of it
X_dependent=X_dependent[:,1:]

#Split the data in to training Set and Test Sets
X_dependent_train,X_dependent_test,y_independent_train,y_independent_test = train_test_split(X_dependent,y_independent,test_size=0.2,random_state=5)

#Run the model with Linear Regression
regressor = LinearRegression()
regressor.fit(X_dependent_train,y_independent_train)

# predict the Analysis with Test varibales of regressor
y_independent_test_predict = regressor.predict(X_dependent_test)

#To add the Ones to first line such that we can use it as x0
X_dependent = np.append(arr=np.ones((50,1)).astype(int),values=X_dependent,axis=1)

#We need to add all the varibale in the X_opt
X_dependent_OPT = X_dependent[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y_independent,exog=X_dependent_OPT).fit()
regressor_OLS.summary()

X_dependent_OPT = X_dependent[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=y_independent,exog=X_dependent_OPT).fit()
regressor_OLS.summary()


X_dependent_OPT = X_dependent[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog=y_independent,exog=X_dependent_OPT).fit()
regressor_OLS.summary()

X_dependent_OPT = X_dependent[:,[0,3,5]]
regressor_OLS = sm.OLS(endog=y_independent,exog=X_dependent_OPT).fit()
regressor_OLS.summary()

X_dependent_OPT = X_dependent[:,[0,3]]
regressor_OLS = sm.OLS(endog=y_independent,exog=X_dependent_OPT).fit()
regressor_OLS.summary()

# ## Automatic Above PRocess
# def backwardElimination(x, sl):
# 	numVars = len(x[0])
# 	for i in range(0, numVars):
# 		regressor_OLS = sm.OLS(y, x).fit()
# 		maxVar = max(regressor_OLS.pvalues).astype(float)
# 		if maxVar > sl:
# 			for j in range(0, numVars - i):
# 				if (regressor_OLS.pvalues[j].astype(float) == maxVar):
# 					x = np.delete(x, j, 1)
# 	regressor_OLS.summary()
# 	return x
#
#
# SL = 0.05
# X_opt = X[:, [0, 1, 2, 3, 4, 5]]
# X_Modeled = backwardElimination(X_opt, SL)