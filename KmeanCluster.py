import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from  sklearn.cluster import KMeans

datasets = pd.read_csv("Kmean/Mall_Customers.csv")

x = datasets.iloc[:,[3,4]].values
# need to Find optimize Cluster by elbow method

wscc=[]
# for i in range(1,11):
# 	kmean = KMeans(n_clusters=i,random_state=0)
# 	kmean.fit(x)
# 	wscc.append(kmean.inertia_)
# plot.plot(range(1,11,1),wscc)
# plot.title("Elbow Method")
# plot.xlabel("Number of Cluster")
# plot.ylabel("WSCC")
# # plot.show()

## By seeing the Elbow graph we can say that 5 is optime Cluster

kmeanOriginal = KMeans(n_clusters=5,random_state=0)
y_predct = kmeanOriginal.fit_predict(x)

## Visiualee the Kmean graph

plot.scatter(x[y_predct==0,0],x[y_predct==0,1],s=100,label="Cluster1")
plot.scatter(x[y_predct==1,0],x[y_predct==1,1],s=100,label="Cluster2")
plot.scatter(x[y_predct==2,0],x[y_predct==2,1],s=100,label="Cluster3")
plot.scatter(x[y_predct==3,0],x[y_predct==3,1],s=100,label="Cluster4")
plot.scatter(x[y_predct==4,0],x[y_predct==4,1],s=100,label="Cluster5")
plot.show(kmeanOriginal.cluster_centers_[:,0],kmeanOriginal.cluster_centers_[:,1],s=200,label="centroid",color='black')
plot.title("Kmean of Spending")
plot.show()