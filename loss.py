import numpy as np 

# COST
def compute_cost(yhat , y , para , lambd = 0):
    logit = -(np.multiply(y , np.log(yhat)) + np.multiply((1-y) , np.log(1 - yhat)))
    """
    theta = para_to_theta(para)
    Using np.linalg.norm(theta)**2 is much faster than np.sum(np.power(theta , 2))
    reg_term = (lambd/2*y.shape[1])*(np.linalg.norm(theta))

    """
    reg_sum = 0
    for i in range(len(para)//2):
        reg_sum += np.linalg.norm(para["w" + str(i+1)] , ord = "fro")

    reg_term = (lambd/2*y.shape[1])*reg_sum

    return np.squeeze(np.sum(logit)/y.shape[1]) + reg_term