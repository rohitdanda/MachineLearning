import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

datasets = pd.read_csv("SVMKernel/Social_Network_Ads.csv")

x = datasets.iloc[:,2:4].values
y = datasets.iloc[:,4].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

classifer = SVC(kernel='rbf')
classifer.fit(x_train,y_train)

y_predct = classifer.predict(x_test)

cm = confusion_matrix(y_test,y_predct)

#visulae the Graph
X , Y = x_train,y_train

X_set, y_set = x_test,y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plot.contourf(X1, X2, classifer.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plot.xlim(X1.min(), X1.max())
plot.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plot.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plot.title('SVM RBF Kernel')
plot.xlabel('Age')
plot.ylabel('Estimated Salary')
plot.legend()
plot.show()