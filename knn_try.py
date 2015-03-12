# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:05:06 2015

@author: pgalland
I/O code from Alexandre Gramfort
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import io



dataset = io.loadmat('data_challenge.mat')

print dataset['X_train'].shape

plt.plot(dataset['X_train'][0])
plt.xlabel('Time')
plt.ylabel('EEG')


print dataset.keys()

X_train, y_train, X_test = dataset['X_train'], dataset['y_train'], dataset['X_test']


labels = np.unique(y_train)
print(labels)


# exemple of dummy classification

print X_train.shape, y_train.shape, X_test.shape

from scipy import stats
XX_train = np.c_[np.std(X_train, axis=1), stats.kurtosis(X_train, axis=1)]
XX_test = np.c_[np.std(X_test, axis=1), stats.kurtosis(X_test, axis=1)]

from sklearn.dummy import DummyClassifier
clf = DummyClassifier()

y_pred = clf.fit(XX_train, y_train).predict(XX_test)

np.savetxt('y_pred.txt', y_pred, fmt='%s')

print X_test.shape
