import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split

#importing the data from CSV file
datasets = pd.read_csv('Data.csv')

#Setting the pre required Data as x and result from them is y
X = datasets.iloc[:, :-1].values
Y = datasets.iloc[:, 3].values

# Here we preprocess the data finding the NAN vlaue and using mean
imputerData = Imputer(missing_values="NaN",strategy="mean",axis=0)
imputerData = imputerData.fit(X[:,1:3])
X[:,1:3] = imputerData.transform(X[:,1:3])

#Encoding the value to binary data

labelEncoder_X = LabelEncoder()
X[:,0]=labelEncoder_X.fit_transform(X[:,0])
labelEncoder_Y = LabelEncoder()
Y = labelEncoder_Y.fit_transform(Y);

# After Label Encoder we have to make Dummy Encoder we use OneHOtEncoder
oneHotEncoder_X = OneHotEncoder(categorical_features = [0])
X = oneHotEncoder_X.fit_transform(X).toarray()

# SPlitting the Data Sets in to Training Sets and Test sets

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=5)

# feature Scaling the data so that values between them wont differ long

X_sandardScaler = StandardScaler()
X_train = X_sandardScaler.fit_transform(X_train)
X_test = X_sandardScaler.transform(X_test)