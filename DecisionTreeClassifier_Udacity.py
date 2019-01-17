from email_preprocess import preprocess
from sklearn import tree
from sklearn.metrics import accuracy_score

features_train, features_test, labels_train, labels_test = preprocess()
len(features_train[0])
clf = tree.DecisionTreeClassifier(min_samples_split=40)
clf.fit(features_train,labels_train)
ypredct = clf.predict(features_test)

accuracy_score(labels_test,ypredct)

print("number of features " , accuracy_score(labels_test,ypredct))



