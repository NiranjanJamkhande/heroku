# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 12:20:56 2021

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

concrete = pd.read_csv("C:\\Users\\Admin\\Downloads\\Concrete_Data.csv")
concrete


X = concrete.iloc[:,0:8]
y = concrete.iloc[:,8]

X.columns

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)


# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))


# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

