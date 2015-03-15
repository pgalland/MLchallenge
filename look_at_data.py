# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 10:26:15 2015

@author: pgalland
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import io


dataset = io.loadmat('data_challenge.mat')
#%%

X_train, y_train, X_test = dataset['X_train'], dataset['y_train'], dataset['X_test']

# number of samples from each class in training set
from collections import Counter
labels, values = zip(*Counter(y_train).items())
indexes = np.arange(len(labels))
width = 1
plt.figure(1)
plt.bar(indexes, values, width)
plt.xticks(indexes + width * 0.5, labels)
plt.show()

# change labels to integers
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(y_train)
y_train_nb = le.transform(y_train)

# idx_labels[i] contains all the indexes of samples of label i
idx_labels = range(5)
for i in range(5):
    idx_labels[i] = np.arange(y_train.size)[y_train_nb == i]

X_train_f = np.fft.rfft(X_train)
X_train_f = np.abs(X_train_f)

plt.figure(2)
for i in range(5):
    for j in range(5):
        plt.subplot(5,5, 5*i+j+1)
        plt.plot(X_train_f[idx_labels[i]][j][:800])
        plt.text(400,5000,"class : "+str(i))
plt.show()
        
