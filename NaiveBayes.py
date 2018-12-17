import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plot
from matplotlib.colors import ListedColormap
import numpy as np

datasets = pd.read_csv("Bayes/Social_Network_Ads.csv")

X = datasets.iloc[:,2:4].values
Y = datasets.iloc[:,4].values

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)

sscaler = StandardScaler()
X_train = sscaler.fit_transform(X_train)
X_test = sscaler.transform(X_test)

classifer = GaussianNB()
classifer.fit(X_train,Y_train)

Y_predct = classifer.predict(X_test)

truth = confusion_matrix(Y_test,Y_predct)

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
plot.title('Navie Bayes (Test set)')
plot.xlabel('Age')
plot.ylabel('Estimated Salary')
plot.legend()
plot.show()