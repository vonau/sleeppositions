#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 16:19:44 2020

@author: nicovonau
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 17:31:58 2020

@author: nicovonau
"""

# Random Forest Classification
import sys #needed on my PC EW
import numpy as np
import pandas as pd
import glob
import time
#EW added some more scores for sure that will also work with the confusion matrix that was used originally in this code
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import sklearn as skl
import matplotlib.pyplot as plt


t1= time.time()
#Importing the dataset
path = 'tempData/features_*.txt' # use your path
all_files = glob.glob(path)

li = []
globLeaveOutLable=[]

for idx, filename in enumerate(all_files):
    
    li.append(pd.read_csv(filename, header= None))
    
    for x in range (len(li[idx])):
        globLeaveOutLable.append(idx)#EW Add subject ID lable needed for leave one out


dataset = pd.concat(li, axis= 0, ignore_index=True, join='inner')

#folder="tempData/"
#dataset = pd.read_csv(folder+"features_18.txt", header=headers)
X = dataset.iloc[:, 0:20].values
y = dataset.iloc[:, 23].values
##normalize force
#force = X[0:6]
#for m in range force:
#    
##normalize force
t=len(X[:,1])
for m in range(t):
    for n in range(6):
        X[m,n]= X[m,n]/np.sum(X[m,0:6])
elapsed1= (time.time()-t1)

# Splitting the dataset into the Training set and Test set
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
# EW: replaced by leave one out crossvalidation
subject_count=range(0,np.amax(globLeaveOutLable)+1)#EW get loop range for leave one out
# Initialize metric matrices
score_acc = np.array([])
score_prec = np.array([])
score_recall = np.array([])
score_f1 = np.array([])
specificity1 = np.array([])
sensitivity1 = np.array([])
roc_auc = np.array([])
y_test_neg= np.array([])
predictions_true= np.array([])

# EW: Loop through subjects with leave one out
for subj_ID in subject_count:
    # Split Training and Test data
    idx_test= []
    idx_train= []
    for i, x in enumerate(globLeaveOutLable):
        if x==subj_ID:
            idx_test.append(i)
        else:
            idx_train.append(i)

    
    X_train = X[idx_train,:]
    # np.disp(len(X_train))
    y_train = y[idx_train]
    X_test  = X[idx_test,:]
    y_test  = y[idx_test]

    # Feature Scaling
#    from sklearn.preprocessing import StandardScaler
#    sc = StandardScaler()
    from sklearn.preprocessing import Normalizer
    sc = Normalizer()
#    from sklearn.preprocessing import RobustScaler
#    sc = RobustScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    t2= time.time()

    # Fitting Random Forest Classification to the Training set
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict_proba(X_test)
    predictions_true = np.append(predictions_true, y_pred[:, 0])
    predictions_false = y_pred[:, 1]
    y_train_neg = y_train*(-1) + 1
    y_test_neg = np.append(y_test_neg, y_test*(-1) + 1)
    
    cut = 0.6
    predictions_hard = (predictions_false > cut).astype(int)
    elapsed2= (time.time()-t2)



    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, predictions_hard)
    # Calculating acc prec recall f1
    score_acc=np.append(score_acc,accuracy_score(y_test, predictions_hard))
    tn, fp, fn, tp = confusion_matrix(y_test, predictions_hard).ravel()

    specificity1= np.append(specificity1,tn/(tn+fp))
    sensitivity1= np.append(sensitivity1,tp/(tp+fn))
    roc_auc = np.append(roc_auc, skl.metrics.roc_auc_score(y_true=y_test, y_score=predictions_false))
    np.disp(score_acc)
np.disp(np.mean(score_acc))
np.disp(np.mean(specificity1))
np.disp(np.mean(sensitivity1))
np.disp(np.std(score_acc))

fpr_nb, tpr_nb, _ = skl.metrics.roc_curve(y_test_neg, predictions_true, pos_label=1)
roc_auc = skl.metrics.roc_auc_score(y_true=y_test, y_score=predictions_false)
print("ROC AUC:", np.mean(roc_auc))

plt.figure(figsize=(15,15))
plt.title('Receiver Operating Characteristic', fontsize=45)
#plt.plot(fpr_rf, tpr_rf,linewidth=4, label= 'Random Forest')
plt.plot(fpr_knn, tpr_knn,linewidth=4,  label= 'KNN')
#plt.plot(fpr_xgb, tpr_xgb,linewidth=4,  label= 'XGBoost')
plt.plot(fpr_nb, tpr_nb,linewidth=4,  label= 'Naive Bayes')
plt.xlabel('False Positive Rate', fontsize= 40)
plt.ylabel('True Positive Rate', fontsize= 40)
plt.legend(loc= 'lower right', fontsize= 40)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)

#print score_acc.mean()
#
#print score_f1.mean()
#print score_prec.mean()

# k-fold cross validation
#from sklearn.model_selection import cross_val_score
#accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
#np.disp(accuracies.mean())
#np.disp(accuracies.std())

#tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

#specificity1= tn/(tn+fp)
#sensitivity1= tp/(tp+fn)
# EW: replaced

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
