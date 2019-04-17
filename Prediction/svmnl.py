# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 00:40:47 2019

@author: Pushkar
"""

import pandas as pd
import numpy as np
import seaborn as sns
from os import path
import matplotlib.pyplot as plt
from wordcloud import WordCloud,ImageColorGenerator,STOPWORDS
import cv2
from sklearn.cluster import KMeans
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.preprocessing import LabelEncoder
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import math

crime_data=pd.read_csv("Chicago_Crimes_2012_to_2017.csv")
crime_data.Date=pd.to_datetime(crime_data.Date,format='%m/%d/%Y %I:%M:%S %p')
crime_data.index=pd.DatetimeIndex(crime_data.Date)
crime_data['time_hour']=crime_data['Date'].apply(lambda x:x.hour)
crime_data['month']=crime_data['Date'].apply(lambda x:x.month)
#crime_data['year']=crime_data['Date'].apply(lambda x:x.year)
crime_data = crime_data[crime_data['Primary Type']=='HOMICIDE']
crime_data = crime_data.dropna()
crime_data.isnull().sum().sum()
keep_cols = ['Arrest','Domestic','District','Location Description','X Coordinate','Y Coordinate','time_hour','month']
crime_data = crime_data[keep_cols].reset_index()
X=crime_data.drop('Arrest',axis=1)
features = list(X.columns)
y = crime_data["Arrest"]
labelencoder = LabelEncoder()
#X['Primary Type']=X['Primary Type'].astype("category").cat.codes
X['Location Description']=X['Location Description'].astype("category").cat.codes
scaler = preprocessing.MinMaxScaler()
X[['X Coordinate', 'Y Coordinate','Location Description','District','time_hour','month']] = scaler.fit_transform(X[['X Coordinate', 'Y Coordinate','Location Description','District','time_hour','month']])
X=X.iloc[:,1:]
Xpred=[0.89,]
# Applying polynomial Kernel SVC
poly_clf = svm.SVC(kernel='rbf', degree=3, C= 50)
X_d = X
y_d = y
# For large values of C, the optimization will choose a smaller-margin hyperplane if that hyperplane does a better job of getting all the training points classified correctly.
# non clean dataset 
poly_accuracy_d = cross_val_score(poly_clf, X_d, y_d, cv=10, scoring='accuracy').mean()
poly_precision_d = cross_val_score(poly_clf, X_d, y_d, cv=10, scoring='precision').mean()
poly_recall_d = cross_val_score(poly_clf, X_d, y_d, cv=10, scoring='recall').mean()

print ('Accuracy for polynomial is', poly_accuracy_d)
print ('Precision for polynomial is', poly_precision_d)
print ('Recall for polynomial is', poly_recall_d)