#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 13:13:23 2019

@author: nicovonau
"""
# Naive Bayes

# Importing the libraries
import numpy as np
import pandas as pd
import time
import glob
import matplotlib.pyplot as plt
import sklearn as skl


#Importing the dataset
path = 'tempData/features_*.txt' # use your path
all_files = glob.glob(path)

li = []
headers= ['pressure sensor 1']
for filename in all_files:
    li.append(pd.read_csv(filename, header= None))

dataset = pd.concat(li, axis= 0, ignore_index=True, join='inner')

#folder="tempData/"
#dataset = pd.read_csv(folder+"features_18.txt", header=headers)
X = dataset.iloc[:, 0:20].values
y = dataset.iloc[:, 23].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

t= time.time()

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB(priors= [0.6, 0.4])
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict_proba(X_test)

elapsed= (time.time()-t)

'''
Evaluation: confusion matrix, k-fold cross validation
'''
predictions_true = y_pred[:, 0]
predictions_false = y_pred[:, 1]
y_train_neg = y_train*(-1) + 1
y_test_neg = y_test*(-1) + 1

cut = 0.6
predictions_hard = (predictions_false > cut).astype(int)

n_bins = 20

plt.figure(figsize=(15,15))
n, bins, patches = plt.hist(predictions_false[y_test == 1], bins=30, alpha=0.5, label= 'positive')
plt.hist(predictions_false[y_test == 0], bins=bins, alpha=0.5, label = 'negative')
plt.xlabel('threshhold', fontsize=40)
plt.legend(loc= 'upper center', fontsize= 40)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.title('Probability Distribution', fontsize=45)
elapsed= (time.time()-t)/60


fpr_nb, tpr_nb, _ = skl.metrics.roc_curve(y_test_neg, predictions_true, pos_label=1)
roc_auc = skl.metrics.roc_auc_score(y_true=y_test, y_score=predictions_false)
print("ROC AUC:", roc_auc)

plt.figure()
plt.plot(fpr_nb, tpr_nb, label = 'KNN')
plt.xlabel('false positives')
plt.ylabel('true positives')
#plt.legend()

## Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions_hard)

## k-fold cross validation
#from sklearn.model_selection import cross_val_score
#accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
#np.disp(accuracies.mean())
#np.disp(accuracies.std())
#
tn, fp, fn, tp = confusion_matrix(y_test, predictions_hard).ravel()

specificity1= tn/(tn+fp)
sensitivity1= tp/(tp+fn)
skl.metrics.accuracy_score(y_test, predictions_hard)
#skl.metrics.classification_report(y_test, predictions_hard)
