# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 18:52:49 2020

@author: Xingy
"""
import numpy as np
def costfunction(X,y,theta):
    m=np.size(X,0)
    predictions=X*theta
    sqrErrors=np.power((predictions-y),2)
    J=1/(2*m)*np.sum(sqrErrors)
    Gradient= (1/m)*np.transpose(X)*(predictions-y)
    return J,Gradient
def gradientdescent(X,y,theta,alpha,iters):
    m=np.size(X,0)
    for i in range(1,iters):
      theta=theta-alpha*(1/m)*np.transpose(X)*(X*theta-y)
    return theta


X=np.mat([[1,1],[1,2],[1,3]])
y=np.mat([[1],[2],[3]])
theta=np.mat([[0],[1]])
[j,G]=costfunction(X,y,theta)
theta1=np.mat([[0],[0]])
[j1,G1]=costfunction(X,y,theta1)



alpha=0.03
iters=1000
initial_theta=np.mat([[0],[0]])
theta2=gradientdescent(X,y,initial_theta,alpha,iters)

