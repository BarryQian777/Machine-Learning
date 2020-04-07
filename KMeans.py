# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 00:36:52 2020

@author: Xingy
"""

from sklearn.datasets import make_blobs
X,y=make_blobs(n_samples=500,n_features=2,centers=4,random_state=1)
import matplotlib.pyplot as plt
fig,ax1=plt.subplots(1)
ax1.scatter(X[:,0],X[:,1]
            ,marker='o'
            ,s=8
            )
plt.show()
from sklearn.cluster import KMeans
n_cluster=3
cluster=KMeans(n_clusters=n_cluster,random_state=0).fit(X)
y_pred=cluster.labels_
pre=cluster.fit_predict(X)
centroid=cluster.cluster_centers_
inertia=cluster.inertia_
color=["red","pink","orange","gray"]
fig,ax1=plt.subplots(1)
for i in range(n_cluster):
    ax1.scatter(X[y_pred==i,0],X[y_pred==i,1]
    ,marker='o'
    ,s=8
    ,c=color[i]
    )
ax1.scatter(centroid[:,0],centroid[:,1]
           ,marker="x"
           ,s=15
           ,c="black"
           )
plt.show()
n_clusters=4
cluster_=KMeans(n_clusters=n_clusters,random_state=0).fit(X)
inertia_=cluster_.inertia_
inertia_   
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
silhouette_score(X,y_pred)  
silhouette_samples()    