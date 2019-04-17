# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 23:26:31 2019

@author: Pushkar
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 22:48:17 2019

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

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
#y=y.iloc[0:10]
# Using GaussianNB
train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size = 0.25, random_state = 42)
gaussian_clf = GaussianNB()
gaussian_clf.fit(train_features, train_labels);
# For large values of C, the optimization will choose a smaller-margin hyperplane if that hyperplane does a better job of getting all the training points classified correctly.
predictions = gaussian_clf.predict(test_features)
# Calculate the absolute errors
results=confusion_matrix(test_labels, predictions)
print(results)
print('Accuracy Score :',accuracy_score(test_labels, predictions)) 
print('Report : ')
print(classification_report(test_labels, predictions))


# Applying 10 fold cross validation
'''gaussian_accuracy = cross_val_score(gaussian_clf, X, y, cv=10).mean()
gaussian_precision= cross_val_score(gaussian_clf, X, y, cv=10, scoring='precision').mean()
gaussian_recall = cross_val_score(gaussian_clf, X, y, cv=10, scoring='recall').mean()
print("Accuracy for gaussian :", gaussian_accuracy)
print("Recall for gaussian:", gaussian_recall)
print("Precision for gaussian:", gaussian_precision)

df_predictiveFeature = crime_data

dictPredFeature = {}
# Collecting data for the two classes
true_df = X[df_predictiveFeature['Arrest'] == 1]
false_df = X[df_predictiveFeature['Arrest'] == 0]
for column in X:
    mean_true = true_df[column].mean()
    mean_false = false_df[column].mean()
    var_true = true_df[column].var()
    var_false = false_df[column].var()
    if(column != 'Arrest'):
        predScore = abs((mean_true - mean_false))/(math.sqrt(var_true)+math.sqrt(var_false))
        dictPredFeature[column] = predScore
most_pred_features = sorted(dictPredFeature.items(), key=lambda x: x[1])[-5:]
for i in most_pred_features:
    print(i)'''

