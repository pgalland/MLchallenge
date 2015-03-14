# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 22:40:42 2015

@author: pgalland
"""
import numpy as np

def cn_to_r2n(a):
    """
    transforms 
    [[1-1j, 2-2j],
     [3-3j, 4-4j]]
    in
    [[1, -1, 2, -2],
     [3, -3, 4, -4]]
    """
    return np.insert(a.imag, np.arange(len(a[0])), a.real, axis=1)

def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


