import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import scipy.cluster.hierarchy as dendro
from sklearn.cluster import AgglomerativeClustering


datasets = pd.read_csv("HierarchicalClustering/Mall_Customers.csv")

X = datasets.iloc[:,[3,4]].values

##Plotting the Dendrogram To find Optimazw the Cluster
# dendroGram = dendro.dendrogram(dendro.linkage(X,method='ward'))
# plot.title("DendroGRam to find Optimze Cluster")
# plot.show() # From the Diagram we get 5 cluster as optimze

clustering = AgglomerativeClustering(n_clusters=5)
y_predct = clustering.fit_predict(X)

plot.scatter(X[y_predct==0,0],X[y_predct==0,1],s=100,label="c1")
plot.scatter(X[y_predct==1,0],X[y_predct==1,1],s=100,label="c2")
plot.scatter(X[y_predct==2,0],X[y_predct==2,1],s=100,label="c3")
plot.scatter(X[y_predct==3,0],X[y_predct==3,1],s=100,label="c4")
plot.scatter(X[y_predct==4,0],X[y_predct==4,1],s=100,label="c5")
plot.title("Agglomerative Clustering")
plot.show()
