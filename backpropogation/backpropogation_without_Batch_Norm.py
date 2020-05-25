import numpy as np 
from activation_functions import d_sigmoid , d_ReLu , d_leaky_ReLu

# Backpropogation
def linear_backward(dz , cache):
    a_prev , w , b = cache
    m = a_prev.shape[1]
    
    dw = np.dot(dz , a_prev.T)/m
    db = np.sum(dz , axis = 1 , keepdims = True)/m
    da_prev = np.dot(w.T , dz)
    
    assert(da_prev.shape == a_prev.shape)
    assert(dw.shape == w.shape)
    assert(db.shape == b.shape)
    
    return da_prev , dw , db

def linear_activation_backward(da , cache , activation):
    linear_cache , activation_cache , dropout_cache = cache
    da = np.multiply(da , dropout_cache)

    if activation == "sigmoid":
        dz = da*d_sigmoid(activation_cache)
        
    elif activation == "ReLu":
        dz = da*d_ReLu(activation_cache)
    
    elif activation == "leaky_ReLu":
        dz = da*d_leaky_ReLu(activation_cache)
    
    da_prev , dw , db = linear_backward(dz , linear_cache)
    return da_prev , dw , db

def l_model_backwards(al , y , caches):
    grad = {}
    l = len(caches)
    y = y.reshape(al.shape)
    
    dal = -(np.divide(y, al) - np.divide(1 - y, 1 - al))
    current_cache = caches[l-1]
    grad["da" + str(l - 1)] , grad["dw" + str(l)] , grad["db" + str(l)] = linear_activation_backward(dal , current_cache , "sigmoid")
    
    for i in range(l-2 , -1 , -1): #we already cal. for last layer above so we start with l-2!
        current_cache = caches[i]
        da_prev_temp , dw_temp , db_temp = linear_activation_backward(grad["da" + str(i+1)] , current_cache , "ReLu")
        grad["da" + str(i)] = da_prev_temp
        grad["dw" + str(i+1)] = dw_temp
        grad["db" + str(i+1)] = db_temp
    
    return grad