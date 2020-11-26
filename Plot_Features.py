#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 10:17:34 2019

@author: nicovonau
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import time

t= time.time()
#Importing the dataset
path = 'tempData/features_*.txt' # use your path
all_files = glob.glob(path)

li = []
headers= ['pressure sensor 1', ]
for filename in all_files:
    li.append(pd.read_csv(filename, header= None))

dataset = pd.concat(li, axis= 0, ignore_index=True, join='inner')

#folder="tempData/"
#dataset = pd.read_csv(folder+"features_18.txt", header=headers)
X = dataset.iloc[:, 0:23].values
y = dataset.iloc[:, 23].values

pressure= dataset.iloc[:, 0:6].values
amp= dataset.iloc[:, 6:12].values
sumdiff= dataset.iloc[:, 12].values
mean_p= dataset.iloc[:, 13].values
skew= dataset.iloc[:, 14].values
kurt= dataset.iloc[:, 15].values
mean_E= dataset.iloc[:, 16].values
skew_E= dataset.iloc[:, 17].values
kurt_E= dataset.iloc[:, 18].values
mean_peak= dataset.iloc[:, 19].values
CoP= dataset.iloc[:, 20].values
CoAE= dataset.iloc[:, 21].values
CoPeak= dataset.iloc[:, 22].values

#8array

plt.figure(figsize=(30,22))
#1sumdiff
plt.subplot2grid((8,8),(0,0))
plt.ylabel('sum of differences [kPa]')
plt.title('sum of differences [kPa]')
plt.scatter(sumdiff, sumdiff)
plt.subplot2grid((8,8),(0,1))
plt.title('mean of pressure vector [kPa]')
plt.scatter(sumdiff, mean_p)
plt.subplot2grid((8,8),(0,2))
plt.title('skewness of pressure vector')
plt.scatter(sumdiff, skew)
plt.subplot2grid((8,8),(0,3))
plt.title('kurtosis of pressure vector')
plt.scatter(sumdiff, kurt)
plt.subplot2grid((8,8),(0,4))
plt.title('mean of energy vector')
plt.scatter(sumdiff, mean_E)
plt.subplot2grid((8,8),(0,5))
plt.title('skewness of energy vector')
plt.scatter(sumdiff, skew_E)
plt.subplot2grid((8,8),(0,6))
plt.title('kurtosis of energy vector')
plt.scatter(sumdiff, kurt_E)
plt.subplot2grid((8,8),(0,7))
plt.title('mean of peaks')
plt.scatter(sumdiff, mean_peak)

#2mean pressure
plt.subplot2grid((8,8),(1,0))
plt.ylabel('mean of pressure vector [kPa]')
plt.scatter(mean_p, sumdiff)
plt.subplot2grid((8,8),(1,1))
plt.scatter(mean_p, mean_p)
plt.subplot2grid((8,8),(1,2))
plt.scatter(mean_p, skew)
plt.subplot2grid((8,8),(1,3))
plt.scatter(mean_p, kurt)
plt.subplot2grid((8,8),(1,4))
plt.scatter(mean_p, mean_E)
plt.subplot2grid((8,8),(1,5))
plt.scatter(mean_p, skew_E)
plt.subplot2grid((8,8),(1,6))
plt.scatter(mean_p, kurt_E)
plt.subplot2grid((8,8),(1,7))
plt.scatter(mean_p, mean_peak)

#3skewness
plt.subplot2grid((8,8),(2,0))
plt.ylabel('skewness of pressure vector')
plt.scatter(skew, sumdiff)
plt.subplot2grid((8,8),(2,1))
plt.scatter(skew, mean_p)
plt.subplot2grid((8,8),(2,2))
plt.scatter(skew, skew)
plt.subplot2grid((8,8),(2,3))
plt.scatter(skew, kurt)
plt.subplot2grid((8,8),(2,4))
plt.scatter(skew, mean_E)
plt.subplot2grid((8,8),(2,5))
plt.scatter(skew, skew_E)
plt.subplot2grid((8,8),(2,6))
plt.scatter(skew, kurt_E)
plt.subplot2grid((8,8),(2,7))
plt.scatter(skew, mean_peak)

#4kurtosis
plt.subplot2grid((8,8),(3,0))
plt.ylabel('kurtosis of pressure vector')
plt.scatter(kurt, sumdiff)
plt.subplot2grid((8,8),(3,1))
plt.scatter(kurt, mean_p)
plt.subplot2grid((8,8),(3,2))
plt.scatter(kurt, skew)
plt.subplot2grid((8,8),(3,3))
plt.scatter(kurt, kurt)
plt.subplot2grid((8,8),(3,4))
plt.scatter(kurt, mean_E)
plt.subplot2grid((8,8),(3,5))
plt.scatter(kurt, skew_E)
plt.subplot2grid((8,8),(3,6))
plt.scatter(kurt, kurt_E)
plt.subplot2grid((8,8),(3,7))
plt.scatter(kurt, mean_peak)

#5mean energy
plt.subplot2grid((8,8),(4,0))
plt.ylabel('mean of energy vector')
plt.scatter(mean_E, sumdiff)
plt.subplot2grid((8,8),(4,1))
plt.scatter(mean_E, mean_p)
plt.subplot2grid((8,8),(4,2))
plt.scatter(mean_E, skew)
plt.subplot2grid((8,8),(4,3))
plt.scatter(mean_E, kurt)
plt.subplot2grid((8,8),(4,4))
plt.scatter(mean_E, mean_E)
plt.subplot2grid((8,8),(4,5))
plt.scatter(mean_E, skew_E)
plt.subplot2grid((8,8),(4,6))
plt.scatter(mean_E, kurt_E)
plt.subplot2grid((8,8),(4,7))
plt.scatter(mean_E, mean_peak)

#6skewness energy
plt.subplot2grid((8,8),(5,0))
plt.ylabel('skewness of energy vector')
plt.scatter(skew_E, sumdiff)
plt.subplot2grid((8,8),(5,1))
plt.scatter(skew_E, mean_p)
plt.subplot2grid((8,8),(5,2))
plt.scatter(skew_E, skew)
plt.subplot2grid((8,8),(5,3))
plt.scatter(skew_E, kurt)
plt.subplot2grid((8,8),(5,4))
plt.scatter(skew_E, mean_E)
plt.subplot2grid((8,8),(5,5))
plt.scatter(skew_E, skew_E)
plt.subplot2grid((8,8),(5,6))
plt.scatter(skew_E, kurt_E)
plt.subplot2grid((8,8),(5,7))
plt.scatter(skew_E, mean_peak)

#7kurtosis energy
plt.subplot2grid((8,8),(6,0))
plt.ylabel('kurtosis of energy vector')
plt.scatter(kurt_E, sumdiff)
plt.subplot2grid((8,8),(6,1))
plt.scatter(kurt_E, mean_p)
plt.subplot2grid((8,8),(6,2))
plt.scatter(kurt_E, skew)
plt.subplot2grid((8,8),(6,3))
plt.scatter(kurt_E, kurt)
plt.subplot2grid((8,8),(6,4))
plt.scatter(kurt_E, mean_E)
plt.subplot2grid((8,8),(6,5))
plt.scatter(kurt_E, skew_E)
plt.subplot2grid((8,8),(6,6))
plt.scatter(kurt_E, kurt_E)
plt.subplot2grid((8,8),(6,7))
plt.scatter(kurt_E, mean_peak)

#8mean peak
plt.subplot2grid((8,8),(7,0))
plt.ylabel('mean peaks')
plt.scatter(mean_peak, sumdiff)
plt.subplot2grid((8,8),(7,1))
plt.scatter(mean_peak, mean_p)
plt.subplot2grid((8,8),(7,2))
plt.scatter(mean_peak, skew)
plt.subplot2grid((8,8),(7,3))
plt.scatter(mean_peak, kurt)
plt.subplot2grid((8,8),(7,4))
plt.scatter(mean_peak, mean_E)
plt.subplot2grid((8,8),(7,5))
plt.scatter(mean_peak, skew_E)
plt.subplot2grid((8,8),(7,6))
plt.scatter(mean_peak, kurt_E)
plt.subplot2grid((8,8),(7,7))
plt.scatter(mean_peak, mean_peak)

#plt.suptitle('Feature vs. Feature')
plt.show()