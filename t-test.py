#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 11:13:41 2019

@author: nicovonau
"""

import scipy.stats as stats
import numpy as np
import pandas as pd
import glob
import time


t= time.time()
## Importing the dataset
path = 'tempData/features_*.txt' # use your path
all_files = glob.glob(path)

li = []
headers= ['pressure sensor 1']
for filename in all_files:
    li.append(pd.read_csv(filename, header= None))

dataset = pd.concat(li, axis= 0, ignore_index=True, join='inner')
X = dataset.iloc[:, 0:20].values
y = dataset.iloc[:, 23].values

D=0
for i in range(len(y)):
    if y[i] == 1:
        D=D+1


sup=np.zeros((D,20))
nonsup=np.zeros((len(X[:,1])-D,20))
DD=-1
Ds=DD
#split data
for i in range(len(y)):
    temp=X[i,:]
    if y[i] == 1:
        DD=DD+1
        sup[DD,:]=temp
    else:
        Ds=Ds+1
        nonsup[Ds,:]=temp

pval=stats.ttest_ind(sup, nonsup).pvalue
six=np.mean(pval[6:12])