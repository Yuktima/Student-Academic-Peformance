# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 20:15:20 2020

@author: YUKTIMA
"""
#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset=pd.read_csv('data.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:,16]

#Label Encoding the dataset
from sklearn.preprocessing import LabelEncoder
le_X=LabelEncoder()
for i in X.iloc[:,:9]:
    X[i]=le_X.fit_transform(X[i])
for i in X.iloc[:,13:16]:
    X[i]=le_X.fit_transform(X[i])

le_y = LabelEncoder()
y=le_y.fit_transform(y)

#Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X ,y ,test_size=0.2,random_state=0)

#Feature Scalaing
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)

#Model Fitting
from sklearn.svm import SVC
model=SVC(kernel='linear',random_state=0)
model.fit(X_train,Y_train)


#Prediction of Dataset
Y_pred=model.predict(X_test)

#Analysis of Model
from sklearn.metrics import accuracy_score
ac=accuracy_score(Y_pred,Y_test)
print(ac*100)

























