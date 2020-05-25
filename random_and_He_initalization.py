import numpy as np

# Initilization
def init_para(layers_dim , init_type = "random" , seed = 0):
    para , l = {} , len(layers_dim)
    np.random.seed(seed)

    if init_type == "random":
        for i in range(1 , l):
            para["w{}".format(i)] = np.random.randn(layers_dim[i] , layers_dim[i-1])*0.01
            para["b{}".format(i)] = np.zeros((layers_dim[i] , 1))
        
            assert(para["w" + str(i)].shape == (layers_dim[i] , layers_dim[i-1]))
            assert(para["b" + str(i)].shape == (layers_dim[i] , 1))

    if init_type == "he":
        for i in range(1 , l):
            para["w{}".format(i)] = np.random.randn(layers_dim[i] , layers_dim[i-1])*np.sqrt(2/layers_dim[i-1])
            para["b{}".format(i)] = np.zeros((layers_dim[i] , 1))
        
            assert(para["w" + str(i)].shape == (layers_dim[i] , layers_dim[i-1]))
            assert(para["b" + str(i)].shape == (layers_dim[i] , 1))

    return para