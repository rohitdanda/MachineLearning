import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#read the data from Csv file
datasets = pd.read_csv("SimpleLinearRegression/Salary_Data.csv")

x = datasets.iloc[:,:-1].values

y = datasets.iloc[:,1].values

#split the data to test and train values

x_train,x_test = train_test_split(x,test_size=0.3,random_state=0)
y_train,y_test = train_test_split(y,test_size=0.3,random_state=0)

## fitting the Linear Regression model to this data

simpleLinear = LinearRegression()
simpleLinear.fit(x_train,y_train)

#predict the value from the model and check it with the test vlaues
y_predct=simpleLinear.predict(x_test)

#Plot the graph with given data and predict data so we can compare
plot.scatter(x_train,y_train,color='red')
plot.plot(x_train,simpleLinear.predict(x_train),color='blue')
plot.title("Differnce of Simple Regression")
plot.xlabel("Experiecne")
plot.ylabel("Salary")
plot.show()

#plot the graph with test data and predict train data

plot.scatter(x_test,y_test,edgecolors='red')
plot.plot(x_train,simpleLinear.predict(x_train,color='blue'))
plot.title("Grpah for Test data with Predict")
plot.xlabel("Experience")
plot.ylabel("Salary")
plot.show()