# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 01:26:42 2020

@author: Prakhar
"""

import pandas as pd
from numpy import *
import numpy as np
from sklearn import preprocessing
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.model_selection import cross_val_score
import pickle

dataset =pd.read_csv(r'C:\Users\Prakhar\Documents\minor project\Personality-prediction-system-master\final project\train_dataset.csv')
X = dataset.iloc[:,:5]
y = dataset.iloc[:,-1]       

    
mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg',max_iter =1000)
mul_lr.fit(X,y)
#saving model to disk
pickle.dump(mul_lr, open('model.pkl', 'wb'))

#model deploy
model= pickle.load(open('model.pkl','rb'))
print(model.predict([[8, 9, 6,9,8]]))