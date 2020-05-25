import numpy as np 

def init_adam(para):
    l = len(para)//2
    v , s= {} , {}

    for i in range(l):
        v["dw" + str(i + 1)] = np.zeros(para["w" + str(i + 1)].shape)
        v["db" + str(i + 1)] = np.zeros(para["b" + str(i + 1)].shape)
        s["dw" + str(i + 1)] = np.zeros(para["w" + str(i + 1)].shape)
        s["db" + str(i + 1)] = np.zeros(para["b" + str(i + 1)].shape)

    return v , s

def init_adam_BN(para):
    l = len(para)//2
    v , s = {} , {}

    for i in range(l - 1):
        v["dw" + str(i + 1)] = np.zeros(para["w" + str(i + 1)].shape)
        v["dgamma" + str(i + 1)] = np.zeros(para["gamma" + str(i + 1)].shape)
        v["dbeta" + str(i + 1)] = np.zeros(para["beta" + str(i + 1)].shape)
        s["dw" + str(i + 1)] = np.zeros(para["w" + str(i + 1)].shape)
        s["dgamma" + str(i + 1)] = np.zeros(para["gamma" + str(i + 1)].shape)
        s["dbeta" + str(i + 1)] = np.zeros(para["beta" + str(i + 1)].shape)
    
    v["dw" + str(l)] = np.zeros(para["w" + str(l)].shape)
    v["dbeta" + str(l)] = np.zeros(para["beta" + str(l)].shape)
    s["dw" + str(l)] = np.zeros(para["w" + str(l)].shape)
    s["dbeta" + str(l)] = np.zeros(para["beta" + str(l)].shapes)

    return v , s

def adam(t , para , grad , v , s , learning_rate = 0.01 , m = 10 , lambd = 0 , beta1 = 0.9 , beta2 = 0.999 ,esplion = 1e-8):
    l = len(para)//2
    v_corr , s_corr = {} , {}

    for i in range(l):
        v["dw" + str(i + 1)] = beta1*v["dw" + str(i + 1)] + (1 - beta1)*grad["dw" + str(i + 1)]
        v["db" + str(i + 1)] = beta1*v["db" + str(i + 1)] + (1 - beta1)*grad["db" + str(i + 1)]

        s["dw" + str(i + 1)] = beta2*s["dw" + str(i + 1)] + (1 - beta2)*np.power(grad["dw" + str(i + 1)] , 2)
        s["db" + str(i + 1)] = beta2*s["db" + str(i + 1)] + (1 - beta2)*np.power(grad["db" + str(i + 1)] , 2)        

        v_corr["dw" + str(i + 1)] = v["dw" + str(i + 1)]/(1 - beta1**t)
        v_corr["db" + str(i + 1)] = v["db" + str(i + 1)]/(1 - beta1**t)

        s_corr["dw" + str(i + 1)] = s["dw" + str(i + 1)]/(1 - beta2**t)
        s_corr["db" + str(i + 1)] = s["db" + str(i + 1)]/(1 - beta2**t)

        para["w" + str(i+1)] = (1 - lambd/m)*para["w" + str(i+1)] - np.divide(learning_rate*v_corr["dw" + str(i+1)] , np.sqrt(s_corr["dw" + str(i + 1)] + esplion))
        para["b" + str(i+1)] = (1 - lambd/m)*para["b" + str(i+1)] - np.divide(learning_rate*v_corr["db" + str(i+1)] , np.sqrt(s_corr["db" + str(i + 1)] + esplion))

    return para , v , s

def adam_with_BN(t , para , grad , v , s , learning_rate = 0.01 , m = 10 , lambd = 0 , beta1 = 0.9 , beta2 = 0.999 ,esplion = 1e-8):
    l = len(para)//2
    v_corr , s_corr = {} , {}

    for i in range(l-1):
        v["dw" + str(i + 1)] = beta1*v["dw" + str(i + 1)] + (1 - beta1)*grad["dw" + str(i + 1)]
        v["dgamma" + str(i + 1)] = beta1*v["dgamma" + str(i + 1)] + (1 - beta1)*grad["dgamma" + str(i + 1)]
        v["dbeta" + str(i + 1)] = beta1*v["dbeta" + str(i + 1)] + (1 - beta1)*grad["dbeta" + str(i + 1)]

        s["dw" + str(i + 1)] = beta2*s["dw" + str(i + 1)] + (1 - beta2)*np.power(grad["dw" + str(i + 1)] , 2)
        s["dgamma" + str(i + 1)] = beta2*s["dgamma" + str(i + 1)] + (1 - beta2)*np.power(grad["dgamma" + str(i + 1)] , 2)        
        s["dbeta" + str(i + 1)] = beta2*s["dbeta" + str(i + 1)] + (1 - beta2)*np.power(grad["dbeta" + str(i + 1)] , 2)

        v_corr["dw" + str(i + 1)] = v["dw" + str(i + 1)]/(1 - beta1**t)
        v_corr["dgamma" + str(i + 1)] = v["dgamma" + str(i + 1)]/(1 - beta1**t)
        v_corr["dbeta" + str(i + 1)] = v["dbeta" + str(i + 1)]/(1 - beta1**t)

        s_corr["dw" + str(i + 1)] = s["dw" + str(i + 1)]/(1 - beta2**t)
        s_corr["dgamma" + str(i + 1)] = s["dgamma" + str(i + 1)]/(1 - beta2**t)
        s_corr["dbeta" + str(i + 1)] = s["dbeta" + str(i + 1)]/(1 - beta2**t)

        para["w" + str(i+1)] = (1 - lambd/m)*para["w" + str(i+1)] - np.divide(learning_rate*v_corr["dw" + str(i+1)] , np.sqrt(s_corr["dw" + str(i + 1)] + esplion))
        para["gamma" + str(i+1)] = (1 - lambd/m)*para["gamma" + str(i+1)] - np.divide(learning_rate*v_corr["dgamma" + str(i+1)] , np.sqrt(s_corr["dgamma" + str(i + 1)] + esplion))
        para["beta" + str(i+1)] = (1 - lambd/m)*para["beta" + str(i+1)] - np.divide(learning_rate*v_corr["dbeta" + str(i+1)] , np.sqrt(s_corr["dbeta" + str(i + 1)] + esplion))

    v["dw" + str(l)] = beta1*v["dw" + str(l)] + (1 - beta1)*grad["dw" + str(l)]
    v["dbeta" + str(l)] = beta1*v["dbeta" + str(l)] + (1 - beta1)*grad["dbeta" + str(l)]

    s["dw" + str(l)] = beta2*s["dw" + str(l)] + (1 - beta2)*np.power(grad["dw" + str(l)] , 2)                
    s["dbeta" + str(l)] = beta2*s["dbeta" + str(l)] + (1 - beta2)*np.power(grad["dbeta" + str(l)] , 2)

    v_corr["dw" + str(l)] = v["dw" + str(l)]/(1 - beta1**t)        
    v_corr["dbeta" + str(l)] = v["dbeta" + str(l)]/(1 - beta1**t)

    s_corr["dw" + str(l)] = s["dw" + str(l)]/(1 - beta2**t)        
    s_corr["dbeta" + str(l)] = s["dbeta" + str(l)]/(1 - beta2**t)

    para["w" + str(l)] = (1 - lambd/m)*para["w" + str(l)] - np.divide(learning_rate*v_corr["dw" + str(l)] , np.sqrt(s_corr["dw" + str(l)] + esplion))
    para["beta" + str(l)] = (1 - lambd/m)*para["beta" + str(l)] - np.divide(learning_rate*v_corr["dbeta" + str(l)] , np.sqrt(s_corr["dbeta" + str(l)] + esplion))

    return para , v , s