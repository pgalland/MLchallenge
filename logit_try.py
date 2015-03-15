# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 19:10:43 2015

@author: pgalland
"""
import numpy as np
from scipy import io


dataset = io.loadmat('data_challenge.mat')

X_train, y_train, X_test = dataset['X_train'], dataset['y_train'], dataset['X_test']

#%%
from functions import (shuffle_in_unison)

shuffle_in_unison(X_train, y_train)

# the Xi are now the modules of the fourier coefficients (we only take 
# the first 1500 ones because not much seems to happen after that)
X_train = np.abs(np.fft.rfft(X_train))[:,:1500]

cut = 8000
X_test1 = X_train[cut:]
y_true1 = y_train[cut:]
X_train = X_train[:cut]
y_train = y_train[:cut]

from sklearn import linear_model
logit_pred = linear_model.LogisticRegression()
logit_pred.fit(X_train,y_train)

y_predicted = logit_pred.predict(X_test1)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_true1, y_predicted))

