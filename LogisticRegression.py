import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
datasets = pd.read_csv("LogisticRegression/Social_Network_Ads.csv")

X_indenpendent = datasets.iloc[:,2:4].values
Y_dependent = datasets.iloc[:,4].values

X_train,X_test,Y_train,Y_test = train_test_split(X_indenpendent,Y_dependent,test_size=0.25,random_state=0)

##feature SCaling

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifer = LogisticRegression(random_state=0)
classifer.fit(X_train,Y_train)
y_predict=classifer.predict(X_test)
cm = confusion_matrix(Y_test,y_predict)

##visulae the Graph
X , Y = X_train,Y_train

X_set, y_set = X_test,Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plot.contourf(X1, X2, classifer.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plot.xlim(X1.min(), X1.max())
plot.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plot.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plot.title('Logistic Regression (Test set)')
plot.xlabel('Age')
plot.ylabel('Estimated Salary')
plot.legend()
plot.show()
	
