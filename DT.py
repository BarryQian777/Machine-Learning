
"""
Created on Wed Mar 25 17:20:17 2020

@author: Xingy
"""

from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
wine=load_wine()
import pandas as pd
pd.concat([pd.DataFrame(wine.data),pd.DataFrame(wine.target)],axis=1)
Xtrain,Xtest,Ytrain,Ytest=train_test_split(wine.data,wine.target,test_size=0.3)
clf=tree.DecisionTreeClassifier(criterion="entropy")
clf=clf.fit(Xtrain,Ytrain)
score=clf.score(Xtest,Ytest)
import graphviz
feature_name=['alcohol',
 'malic_acid',
 'ash',
 'alcalinity_of_ash',
 'magnesium',
 'total_phenols',
 'flavanoids',
 'nonflavanoid_phenols',
 'proanthocyanins',
 'color_intensity',
 'hue',
 'od280/od315_of_diluted_wines',
 'proline']
dot_data=tree.export_graphviz(clf
,feature_names=feature_name 
,class_names=["琴酒","雪莉","贝尔摩德"]
,filled=True
,rounded=True)
graph=graphviz.Source(dot_data)
[*zip(feature_name,clf.feature_importances_)]