import numpy as np

# Activation functions and their Derivatives
def sigmoid(n):
    return 1/(1 + np.exp(-n)) , n

def d_sigmoid(n):
    s = 1/(1 + np.exp(-n))
    return s*(1 - s)

def d_leaky_ReLu(n):
    n = np.where(n > 0 , 1 , 0.001)
    return n

def leaky_ReLu(n):
    n_new = np.maximum(0.001*n , n)
    assert(n_new.shape == n.shape)
    cache = n 
    return n_new , cache

def d_ReLu(n):
    n = np.where(n > 0 , 1 , 0)
    return n

def ReLu(n):
    n_new = np.maximum(0 , n)
    assert(n_new.shape == n.shape)
    cache = n 
    return n_new , cache