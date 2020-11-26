#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:05:58 2019

@author: nicovonau
"""

import numpy as np
import pandas as pd
import glob
import time


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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# Importing the dataset
t= time.time()
# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier(gamma=1, max_depth=5, colsample_bytree =0.6, subsample=0.5, min_child_weight= 3, learning_rate= 0.2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

#elapsed time to train model
elapsed= (time.time()-t)

'''
Evaluation: confusion matrix, k-fold cross validation
'''
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
np.disp(accuracies.mean())
np.disp(accuracies.std())

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

specificity1= tn/(tn+fp)
sensitivity1= tp/(tp+fn)

# Applying Grid Search to find the best model and the best parameters
#from sklearn.model_selection import GridSearchCV
#parameters = [{'min_child_weight': [3],'gamma': [0.5, 1],'subsample': [0.6,0.5],'colsample_bytree': [0.6],'max_depth': [5], 'learning_rate': [0.3,0.2]}]
#grid_search = GridSearchCV(estimator = classifier,
#                           param_grid = parameters,
#                           scoring = 'roc_auc',
#                           cv = 3,
#                           n_jobs = -1)
#grid_search = grid_search.fit(X_train, y_train)
#best_accuracy = grid_search.best_score_
#best_parameters = grid_search.best_params_
####