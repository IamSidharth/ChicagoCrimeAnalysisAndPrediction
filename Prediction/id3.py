# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 23:58:27 2019

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

y = crime_data["Arrest"]
labelencoder = LabelEncoder()
#X['Primary Type']=X['Primary Type'].astype("category").cat.codes
X['Location Description']=X['Location Description'].astype("category").cat.codes
scaler = preprocessing.MinMaxScaler()
X[['X Coordinate', 'Y Coordinate','Location Description','District','time_hour','month']] = scaler.fit_transform(X[['X Coordinate', 'Y Coordinate','Location Description','District','time_hour','month']])
X=X.iloc[:,1:]
features = list(X.columns)
crime_data=crime_data.iloc[:,1:]
#y=y.iloc[0:10]

dt_clf = DecisionTreeClassifier()
dt_clf.fit(X,y)
predicted = dt_clf.predict(X[features])
recall_score = metrics.recall_score(crime_data['Arrest'], predicted)
precision_score = metrics.precision_score(crime_data['Arrest'], predicted)
accuracy_score = metrics.accuracy_score(crime_data['Arrest'], predicted)
print("Training Accuracy = {} Precision = {} Recall = {}".format(accuracy_score,precision_score,recall_score))


for depth in range(1,10):
    tree_clf = DecisionTreeClassifier(max_depth=depth, random_state=0)
    if tree_clf.fit(X,y).tree_.max_depth < depth:
        break
    score = np.mean(cross_val_score(tree_clf, X, y,scoring='accuracy', cv=10, n_jobs=1))
    print("Depth: %i Accuracy: %.3f" % (depth,score))


    

dt_clf = DecisionTreeClassifier(max_depth=2)
dt_clf.fit(X,y)
predicted = dt_clf.predict(X[features])
dt_recall = metrics.recall_score(crime_data['Arrest'], predicted)
dt_precision = metrics.precision_score(crime_data['Arrest'], predicted)
dt_accuracy= metrics.accuracy_score(crime_data['Arrest'], predicted)
print("Accuracy for DT =",dt_accuracy)
print("Precision for DT =",dt_precision)
print("Recall for DT =",dt_precision)


importances=dt_clf.feature_importances_
indices = np.argsort(importances)[::-1]
# Print the feature ranking
print("Feature ranking:")
# We have taken the top 5 feature 
print("The main features used for classification")
print(X.columns[indices[:5]])
print("Top main feature is",X.columns[indices[:1]][0])