import numpy as np
from activation_functions import sigmoid , ReLu , leaky_ReLu

# Forward Model
def linear_forward(a , w , b):
    z = np.dot(w , a) + b
    assert(z.shape == (w.shape[0] , a.shape[1]))
    cache = (a , w , b)
    return z , cache

def linear_activation_forward(a_prev , w , b , activation = "sigmoid" , keep_prob = 1.0):
    
    if activation == "sigmoid":
        z , linear_cache = linear_forward(a_prev , w , b)
        a , activation_cache = sigmoid(z)
    elif activation == "ReLu":
        z , linear_cache = linear_forward(a_prev , w , b)
        a , activation_cache = ReLu(z)
    elif activation == "leaky_ReLu":
        z , linear_cache = linear_forward(a_prev , w , b)
        a , activation_cache = leaky_ReLu(z)

    # Dropout Regularization
    d = np.random.rand(a.shape[0] , a.shape[1])
    d = np.where(d < keep_prob , 1 , 0) # if keep_prob = 1 - no effect of dropout!
    a = np.multiply(a , d)

    assert(a.shape == (w.shape[0] , a_prev.shape[1]))
    #cache of layer l contains ((a[l-1] , w[l] , b[l]) , z[l])
    cache = (linear_cache , activation_cache , d)
    return a , cache

def l_forward_model(X , para , keep_prob = 1.0):
    l = len(para)//2
    a , caches = X , []
    
    for i in range(1 , l):
        a_prev = a 
        a , cache = linear_activation_forward(a_prev , para["w" + str(i)] , para["b" + str(i)] , "ReLu" , keep_prob)
        caches.append(cache)
    
    al , cache = linear_activation_forward(a , para["w" + str(l)] , para["b" + str(l)] , "sigmoid" , keep_prob = 1.0)
    caches.append(cache)
    assert(al.shape == (1 , X.shape[1]))
    return al , caches