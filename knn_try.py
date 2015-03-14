# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:05:06 2015

@author: pgalland
challenge organized by from Alexandre Gramfort
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import io


dataset = io.loadmat('data_challenge.mat')

X_train, y_train, X_test = dataset['X_train'], dataset['y_train'], dataset['X_test']

#%%
from functions import (shuffle_in_unison, cn_to_r2n)

shuffle_in_unison(X_train, y_train)

X_small = X_train[:128]
y_small = y_train[:128]
X_large = X_train[128:]
y_large = y_train[128:]

X_small_f = np.fft.rfft(X_small)
X_large_f = np.fft.rfft(X_large)

X_small_fr = cn_to_r2n(X_small_f)
X_large_fr = cn_to_r2n(X_large_f)

#%%






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
