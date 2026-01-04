import numpy as np

def compute_soft_masks(e1,e2,p=1.0,epsilon=1e-10):
    # compute wiener style soft time frequency masks for 2 sources
    if e1.shape != e2.shape:
        raise ValueError("Energy maps E1 and E2 must have same shape")
    
    if p<=0:
        raise ValueError("Mask exponent p must be +ve")
    
    e1_p = np.power(e1,p)
    e2_p = np.power(e2,p)
    denominator = e1_p + e2_p + epsilon

    m1 = e1_p/denominator
    m2 = e2_p/denominator

    return m1,m2
