# ChicagoCrimeAnalysisAndPrediction

Data set is taken from KaggleList.

First all parameters were analysed, correlation between the attributes were figured out.

Clustering was done by k-means algorithm as the data set is huge (nearly 6M entries), and as k-means is the fastest algorithm.
By default the missing values were taken to be zero in this.

Predicting whether an arrest would be made or not for the given
parameters of crime type Homicide.
The entries with missing values were dropped in this algo.
Predicing algorithms with 72% accuracy were coded 
