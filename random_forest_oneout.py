#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 18:41:31 2019

@author: nicovonau
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 22:28:19 2019

@author: nicovonau
"""
# Random Forest Classification

import numpy as np
import pandas as pd
import glob
import time


t1= time.time()
#Importing the dataset
path = 'tempData/features_*.txt' # use your path
all_files = glob.glob(path)

li = []
ts = []
for filename in all_files:
    if filename == 'tempData/features_21.txt' or filename == 'tempData/features_22.txt' or filename == 'tempData/features_18.txt' :
        ts.append(pd.read_csv(filename, header= None))
    else:
        li.append(pd.read_csv(filename, header= None))

datasettrain = pd.concat(li, axis= 0, ignore_index=True, join='inner')
datasettest = pd.concat(ts, axis= 0, ignore_index=True, join='inner')

#folder="tempData/"
#dataset = pd.read_csv(folder+"features_18.txt", header=headers)
X_train = datasettrain.iloc[:, 0:20].values
y_train = datasettrain.iloc[:, 23].values

X_test = datasettest.iloc[:, 0:20].values
y_test = datasettest.iloc[:, 23].values

elapsed1= (time.time()-t1)

# Splitting the dataset into the Training set and Test set
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

t2= time.time()

# Fitting Random Forest Classification to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier(gamma=1, max_depth=5, colsample_bytree =0.6, subsample=0.5, min_child_weight= 3, learning_rate= 0.2)
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

elapsed2= (time.time()-t2)

'''
Evaluation: confusion matrix, k-fold cross validation
'''

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# k-fold cross validation
#from sklearn.model_selection import cross_val_score
#accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
#np.disp(accuracies.mean())
#np.disp(accuracies.std())

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

specificity1= tn/(tn+fp)
sensitivity1= tp/(tp+fn)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
np.disp(acc)

'''
Feature Selection
'''
#from sklearn.feature_selection import RFE
#selector = RFE(classifier,5, step=1)
#selector = selector.fit(X_train, y_train)
#selector.support_
#selector.ranking_

# Applying Grid Search to find the best model and the best parameters
#from sklearn.model_selection import GridSearchCV
#parameters = [{'n_estimators': [150, 200], 'criterion': ['gini','entropy'], 'class_weight': ['balanced_subsample', 'balanced']},
#             ]
#grid_search = GridSearchCV(estimator = classifier,
#                           param_grid = parameters,
#                           scoring = 'roc_auc',
#                           cv = 3,
#                           n_jobs = -1)
#grid_search = grid_search.fit(X_train, y_train)
#best_accuracy = grid_search.best_score_
#best_parameters = grid_search.best_params_

# plots
#import pylab as pl
#pl.matshow(cm)
#pl.title('Confusion matrix')
#pl.colorbar()
#pl.ylabel('True label')
#pl.xlabel('Predicted label')
#pl.show()
#
#
#t= []