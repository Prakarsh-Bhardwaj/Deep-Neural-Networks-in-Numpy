import numpy as np

def init_velocity(para):
    l = len(para)//2
    v = {}

    for i in range(l):
        v["dw" + str(i + 1)] = np.zeros(para["w" + str(i + 1)].shape)
        v["db" + str(i + 1)] = np.zeros(para["b" + str(i + 1)].shape)
    
    return v

def momentum(para , grad , v , learning_rate , m = 10 , lambd = 0 , beta = 0.9):
    l = len(para)//2

    for i in range(l):
        v["dw" + str(i + 1)] = beta*v["dw" + str(i + 1)] + (1 - beta)*grad["dw" + str(i + 1)]
        v["db" + str(i + 1)] = beta*v["db" + str(i + 1)] + (1 - beta)*grad["db" + str(i + 1)]

        para["w" + str(i+1)] = (1 - lambd/m)*para["w" + str(i+1)] - learning_rate*v["dw" + str(i+1)]
        para["b" + str(i+1)] = (1 - lambd/m)*para["b" + str(i+1)] - learning_rate*v["db" + str(i+1)]
    
    return para , v

def init_velocity_BN(para):
    l = len(para)//2
    v = {}

    for i in range(l - 1):
        v["dw" + str(i + 1)] = np.zeros(para["w" + str(i + 1)].shape)
        v["dgamma" + str(i + 1)] = np.zeros(para["gamma" + str(i + 1)].shape)
        v["dbeta" + str(i + 1)] = np.zeros(para["beta" + str(i + 1)].shape)
    
    v["dw" + str(l)] = np.zeros(para["w" + str(l)].shape)
    v["dbeta" + str(l)] = np.zeros(para["beta" + str(l)].shape)

    return v

def momentum_with_BN(para , grad , v , learning_rate , m = 10 , lambd = 0 , beta = 0.9):
    l = len(para)//2

    for i in range(l-1):
        v["dw" + str(i + 1)] = beta*v["dw" + str(i + 1)] + (1 - beta)*grad["dw" + str(i + 1)]
        v["dgamma" + str(i + 1)] = beta*v["dgamma" + str(i + 1)] + (1 - beta)*grad["dgamma" + str(i + 1)]
        v["dbeta" + str(i + 1)] = beta*v["dbeta" + str(i + 1)] + (1 - beta)*grad["dbeta" + str(i + 1)]

        para["w" + str(i+1)] = (1 - lambd/m)*para["w" + str(i+1)] - learning_rate*v["dw" + str(i+1)]
        para["gamma" + str(i+1)] = (1 - lambd/m)*para["gamma" + str(i+1)] - learning_rate*v["dgamma" + str(i+1)]
        para["beta" + str(i+1)] = (1 - lambd/m)*para["beta" + str(i+1)] - learning_rate*v["dbeta" + str(i+1)]
    
    v["dw" + str(l)] = beta*v["dw" + str(l)] + (1 - beta)*grad["dw" + str(l)]
    v["dbeta" + str(l)] = beta*v["dbeta" + str(i + 1)] + (1 - beta)*grad["dbeta" + str(l)]

    para["w" + str(l)] = (1 - lambd/m)*para["w" + str(i+1)] - learning_rate*v["dw" + str(l)]
    para["beta" + str(l)] = (1 - lambd/m)*para["beta" + str(l)] - learning_rate*v["dbeta" + str(l)]
    
    return para , v