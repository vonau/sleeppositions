#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 22:28:19 2019

@author: nicovonau
"""

#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
#Importing the dataset
path = 'tempData/features_*.txt' # use your path
all_files = glob.glob(path)

li = []
headers= pd.DataFrame([i for i in range(24)])
for filename in all_files:
    li.append(pd.read_csv(filename, header= None))

dataset = pd.concat(li, axis= 0, ignore_index=True, join='inner')

#folder="tempData/"
#dataset = pd.read_csv(folder+"features_18.txt", header=headers)
X = dataset.iloc[:, 12:20].values
y = dataset.iloc[:, 23].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Loading the dataset
df = pd.DataFrame(X)
df.head()


labels= ['sumdiff', 'mean of pressure vector', 'skewness of pressure vector', 'kurtosis of pressure vector', 'mean of energy vector', 'skewness of energy vector', 'kurtosis of energy vector', 'mean of peaks vector']
labels3= ['sumdiff', 'meanP', 'skewP', 'kurtP', 'meanE', 'skewE', 'kurtE', 'mean peaks']
labels2= ['pressure S1', 'pressure S2','pressure S3', 'pressure S4','pressure S5', 'pressure S6','energy S1','energy S2','energy S3','energy S4','energy S5','energy S6','diff. sum', 'mean p.', 'skew. p', 'kurt. p.', 'mean e.', 'skew. e.', 'kurt. e.', 'mean peaks']

#Using Pearson Correlation
plt.figure(figsize=(15,13))
cor = df.corr()
#sns.heatmap(cor, annot=False, cmap=plt.cm.Reds, center=0)
sns.heatmap(cor, annot=True, cmap="Blues")
plt.title('Correlation of Features', fontsize=45)
plt.xticks(np.arange(8), labels=labels3, rotation=30, fontsize=30)
plt.yticks(np.arange(8)+0.5, labels=labels3, rotation='horizontal', fontsize=30)
plt.show()