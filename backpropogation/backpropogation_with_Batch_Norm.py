import numpy as np 
from activation_functions import d_sigmoid , d_ReLu , d_leaky_ReLu
from backpropogation_without_Batch_Norm import linear_activation_backward

# Backpropogation with Batch Normalization
def linear_backward_BN(dz , cache):
    a_prev , w , gamma , beta , normalization_cache = cache
    m = a_prev.shape[1]
    
    dgamma = np.sum(np.multiply(dz , normalization_cache) , axis= 1 , keepdims = True)/m
    dbeta = np.sum(dz , axis = 1 , keepdims = True)/m
    dz_unnormalized = np.multiply(dz , gamma)
    dw = np.dot(dz_unnormalized , a_prev.T)/m

    da_prev = np.dot(w.T , dz_unnormalized)
    
    assert(da_prev.shape == a_prev.shape)
    assert(dw.shape == w.shape)
    assert(dgamma.shape == gamma.shape)
    assert(dbeta.shape == beta.shape)
    
    return da_prev , dw , dgamma , dbeta

def linear_activation_backward_BN(da , cache , activation):
    linear_BN_cache , activation_cache , dropout_cache = cache
    da = np.multiply(da , dropout_cache)

    if activation == "sigmoid":
        dz = da*d_sigmoid(activation_cache)
        
    elif activation == "ReLu":
        dz = da*d_ReLu(activation_cache)
    
    elif activation == "leaky_ReLu":
        dz = da*d_leaky_ReLu(activation_cache)
    
    da_prev , dw , dgamma , dbeta = linear_backward_BN(dz , linear_BN_cache)

    return da_prev , dw , dgamma , dbeta

def l_model_backwards_BN(al , y , caches):
    grad = {}
    l = len(caches)
    y = y.reshape(al.shape)
    
    dal = -(np.divide(y, al) - np.divide(1 - y, 1 - al))
    current_cache = caches[l-1]

    # Scince we didn't do batch norm for output layer!
    # Note - The cache of output layer don't has any norm cache as we did linear act. without BN.
    grad["da" + str(l - 1)] , grad["dw" + str(l)] , grad["dbeta" + str(l)] = linear_activation_backward(dal , current_cache , "sigmoid")
    
    for i in range(l-2 , -1 , -1): #we already cal. for last layer above so we start with l-2!
        current_cache = caches[i]
        da_prev_temp , dw_temp , dgamma_temp , dbeta_temp = linear_activation_backward_BN(grad["da" + str(i+1)] , current_cache , "ReLu")
        grad["da" + str(i)] = da_prev_temp
        grad["dw" + str(i+1)] = dw_temp
        grad["dgamma" + str(i+1)] = dgamma_temp
        grad["dbeta" + str(i+1)] = dbeta_temp
    
    return grad