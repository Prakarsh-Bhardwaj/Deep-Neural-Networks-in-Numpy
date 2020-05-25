import numpy as np
from activation_functions import sigmoid , ReLu , leaky_ReLu
from forward_propogation_without_Batch_Norm import linear_activation_forward

# Forward model with Batch Normalization
def normalize(z , gamma , beta , epslion = 1e-8):

    normalization_cache = z
    mean = np.mean(z , axis = 1)
    var = np.var(z , axis = 1)
    z = (z - mean)/np.sqrt(var + epslion)
    zbar = gamma*z + beta

    return zbar , normalization_cache

def linear_forward_BN(a , w , gamma , beta):

    z = np.dot(w , a)
    z , normalization_cache = normalize(z , gamma , beta)

    assert(z.shape == (w.shape[0] , a.shape[1]))
    linear_cache = (a , w , gamma , beta)
    cache = (linear_cache , normalization_cache)

    return z , cache

def linear_activation_forward_BN(a_prev , w , gamma , beta , activation = "sigmoid" , keep_prob = 1.0):
    
    if activation == "sigmoid":
        z , linear_BN_cache = linear_forward_BN(a_prev , w , gamma , beta)
        a , activation_cache = sigmoid(z)

    elif activation == "ReLu":
        z , linear_BN_cache = linear_forward_BN(a_prev , w , gamma , beta)
        a , activation_cache = ReLu(z)

    elif activation == "leaky_ReLu":
        z , linear_BN_cache = linear_forward_BN(a_prev , w , gamma , beta)
        a , activation_cache = leaky_ReLu(z)

    # Dropout Regularization
    d = np.random.rand(a.shape[0] , a.shape[1])
    d = np.where(d < keep_prob , 1 , 0) # if keep_prob = 1 - no effect of dropout!
    a = np.multiply(a , d)

    assert(a.shape == (w.shape[0] , a_prev.shape[1]))
    #cache of layer l contains ((a[l-1] , w[l] , b[l]) , z[l])
    cache = (linear_BN_cache , activation_cache , d)
    return a , cache

def l_forward_model_BN(X , para , keep_prob = 1.0 , epslion = 1e-8):

    l = len(para)//2
    a , caches = X , []
    
    for i in range(1 , l):
        a_prev = a 
        a , cache = linear_activation_forward_BN(a_prev , para["w" + str(i)] , para["gamma" + str(i)] , para["beta" + str(i)] , "ReLu" , keep_prob)
        caches.append(cache)
    
    # don't do BN for output layer!
    # Note - The cache of output layer don't has any norm cache as we did linear act. without BN.
    al , cache = linear_activation_forward(a , para["w" + str(l)] , para["beta" + str(l)] , "sigmoid" , keep_prob = 1.0)
    caches.append(cache)

    assert(al.shape == (1 , X.shape[1]))
    return al , caches