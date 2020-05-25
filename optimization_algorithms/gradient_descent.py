import numpy as np

def gradient_descent(para , grad , learning_rate , m = 10 , lambd = 0):
    l = len(para)//2

    for i in range(l):
        para["w" + str(i+1)] = (1 - lambd/m)*para["w" + str(i+1)] - learning_rate*grad["dw" + str(i+1)]
        para["b" + str(i+1)] = (1 - lambd/m)*para["b" + str(i+1)] - learning_rate*grad["db" + str(i+1)]
    
    return para

def gradient_descent_with_BN(para , grad , learning_rate , m = 10 , lambd = 0):
    l = len(para)//2

    for i in range(l - 1):
        para["w" + str(i+1)] = (1 - lambd/m)*para["w" + str(i+1)] - learning_rate*grad["dw" + str(i+1)]
        para["gamma" + str(i+1)] = (1 - lambd/m)*para["gamma" + str(i+1)] - learning_rate*grad["dgamma" + str(i+1)]
        para["beta" + str(i+1)] = (1 - lambd/m)*para["beta" + str(i+1)] - learning_rate*grad["dbeta" + str(i+1)]
    
    para["w" + str(l)] = (1 - lambd/m)*para["w" + str(l)] - learning_rate*grad["dw" + str(l)]
    para["beta" + str(l)] = (1 - lambd/m)*para["beta" + str(l)] - learning_rate*grad["dbeta" + str(l)]

    return para