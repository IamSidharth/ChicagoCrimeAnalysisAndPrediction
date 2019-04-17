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
from mlxtend.plotting import plot_decision_regions

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
#X1=pd.dataframe([False,	0.29166666666666663	0.875	0.5084647897082935	0.29458779706255456	0.9130434782608695	0.36363636363636365])


linearsvmclf = svm.SVC(kernel='linear', C =1.0)
accuracy_svm = cross_val_score(linearsvmclf, X, y, cv=10, scoring='accuracy').mean()
precision_svm = cross_val_score(linearsvmclf, X, y, cv=10, scoring='precision').mean()
recall_svm = cross_val_score(linearsvmclf, X, y, cv=10, scoring='recall').mean()
print ('Accuracy for LinearSVC is', accuracy_svm)
print ('Precision for LinearSVC is', precision_svm)
print ('Recall for LinearSVC is', recall_svm)

y=y.astype(np.integer)
plot_decision_regions(X=X.values, 
                      y=y.values,
                      clf=linearsvmclf, 
                      legend=2)