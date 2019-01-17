from time import time

from email_preprocess import preprocess


from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier

features_train, features_test, labels_train, labels_test = preprocess()
# t0=time()
# print("Here we clacluate the KNN")
# clf_Knn = KNeighborsClassifier(n_neighbors=3)
# clf_Knn.fit(features_train,labels_train)
# print ("training time:", round(time()-t0, 3), "s")
# t1=time()
# yKnn_predict=clf_Knn.predict(features_test)
# print ("predict time:", round(time()-t1, 3), "s")
# print("Accuracy of KNN: ",accuracy_score(labels_test,yKnn_predict))

# t0=time()
# print("Here we clacluate the RandomForest")
# clf_Random = RandomForestClassifier()
# clf_Random.fit(features_train,labels_train)
# print ("training time:", round(time()-t0, 3), "s")
# t1=time()
# yKnn_predict=clf_Random.predict(features_test)
# print ("predict time:", round(time()-t1, 3), "s")
# print("Accuracy of RandomForest: ",accuracy_score(labels_test,yKnn_predict))t0=time()

t0=time()
print("Here we clacluate the AdaBoost")
clf_Boost = AdaBoostClassifier()
clf_Boost.fit(features_train,labels_train)
print ("training time:", round(time()-t0, 3), "s")
t1=time()
yBoost_predict=clf_Boost.predict(features_test)
print ("predict time:", round(time()-t1, 3), "s")
print("Accuracy of RandomForest: ",accuracy_score(labels_test,yBoost_predict))
