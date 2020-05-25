import numpy as np 

def para_to_theta(para):
    theta = np.array([])

    for key in para.keys():
        temp = para[key].ravel()
        theta = np.concatenate((theta , temp) , axis = 0)
    
    return theta

def theta_to_para(theta , layers_dim):

    k , para = 0 , {}
    for i in range(1 , len(layers_dim)):

        para["w{}".format(i)] = theta[k : k + layers_dim[i]*layers_dim[i-1]].reshape(layers_dim[i] , layers_dim[i-1])
        k += layers_dim[i]*layers_dim[i-1]
        para["b{}".format(i)] = theta[k : k + layers_dim[i]].reshape(layers_dim[i] , 1)
        k += layers_dim[i]
        
        assert(para["w" + str(i)].shape == (layers_dim[i] , layers_dim[i-1]))
        assert(para["b" + str(i)].shape == (layers_dim[i] , 1))

    return para

def gradient_checking(grad , X , y , para , layers_dim , esplion = 10^(-7) , lambd = 0):
    theta = para_to_theta(para)
    d_theta = para_to_theta(grad)
    Jplus , Jminus = np.zeros(theta.shape) , np.zeros(theta.shape)

    for i in range(len(theta)):
        theta_plus , theta_minus = np.copy(theta) , np.copy(theta)
        theta_plus[i] += esplion
        theta_minus[i] -= esplion

        para_plus , para_minus = theta_to_para(theta_plus , layers_dim) , theta_to_para(theta_minus , layers_dim)

        # regularization is not considered during grad check as we are only checking if our backprop works correctly.
        a_plus , _ = l_forward_model(X , para_plus)
        a_minus , _ = l_forward_model(X , para_minus)
        Jplus[i] , Jminus[i] = compute_cost(a_plus , y , para_plus) , compute_cost(a_minus , y , para_minus)

    grad_approx = (Jplus - Jminus)/(2*esplion)

    return np.linalg.norm(grad_approx - d_theta)/(np.linalg.norm(grad_approx) + np.linalg.norm(d_theta))