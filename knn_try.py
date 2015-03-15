# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:05:06 2015

@author: pgalland
challenge organized by from Alexandre Gramfort
"""
import numpy as np
from scipy import io


dataset = io.loadmat('data_challenge.mat')

X_train, y_train, X_test = dataset['X_train'], dataset['y_train'], dataset['X_test']

#%%
from functions import (shuffle_in_unison)

shuffle_in_unison(X_train, y_train)

X_small = X_train[:512]
y_small = y_train[:512]
X_large = X_train[512:]
y_large = y_train[512:]

X_small_f = np.fft.rfft(X_small)
X_large_f = np.fft.rfft(X_large)




#%%
from sklearn import neighbors
from sklearn.metrics import accuracy_score

knn_param=5
myKNN = neighbors.KNeighborsClassifier(n_neighbors=knn_param)
myKNN.fit(X_small, y_small)

y_predicted = myKNN.predict(X_large)
print(accuracy_score(y_large, y_predicted))






#%%
# exemple of dummy classification
"""
from scipy import stats
XX_train = np.c_[np.std(X_train, axis=1), stats.kurtosis(X_train, axis=1)]
XX_test = np.c_[np.std(X_test, axis=1), stats.kurtosis(X_test, axis=1)]

from sklearn.dummy import DummyClassifier
clf = DummyClassifier()

y_pred = clf.fit(XX_train, y_train).predict(XX_test)

np.savetxt('y_pred.txt', y_pred, fmt='%s')
"""
#print X_test.shape
