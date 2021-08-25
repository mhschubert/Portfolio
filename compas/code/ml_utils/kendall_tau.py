from scipy.stats import rankdata
import numpy as np

def calc_weighting(ranking):
    #hyperbolic weighting
    
    weighting = 1/(1+ranking)
    #print(weighting)
    return weighting

def normalize(array):
    array = np.absolute(array)
    array = array/np.sum(array)
    return array
    
    
def weighted_kendall_tau_distance(x, y, weigh = False, weighting=None):
    
    x = x.reshape((1,-1))
    y = y.reshape((1,-1))
    
    x = normalize(x)
    y = normalize(y)
    x = rankdata(x, method='min')
    y = rankdata(y, method='min')
    
    #bring ranks in ascending order for x and arrange the two arrays in the same order
    perm = np.argsort(x, kind='mergesort')
    x, y = x[perm].reshape((1,-1)), y[perm].reshape((1,-1))
    
    
    assert x.shape == y.shape
    
    if weigh and not weighting:
        weighting_x = calc_weighting(x)
        weighting_y = calc_weighting(y) 
        
    
    
    #is static matrix of shape (x.shape[1],x.shape[1]) with upper triangle == False as we sorted by x
    #x_t = x.reshape((-1,1))
    #comp_X = np.greater(x_t, x)

    comp_X = np.full((x.shape[1], x.shape[1]), True)
    comp_X[np.triu_indices(comp_X.shape[0])] = False
    
    y_t = y.reshape((-1,1)) 
    comp_Y = np.greater(y_t, y)
    comp_Y[np.triu_indices(comp_Y.shape[0])] = False
    comp = np.logical_xor(comp_X, comp_Y)
    
    if weigh:
        
        #calc weighting matrix according to Imbrahim et al. 2019 Global Explanations
        weighted_x = np.multiply(x, weighting_x)
        weighted_y = np.multiply(y, weighting_y)
        weighted_X = np.multiply(weighted_x.reshape((-1,1)), weighted_x)
        weighted_Y = np.multiply(weighted_y.reshape((-1,1)), weighted_y)
        Weighting = np.multiply(weighted_X, weighted_Y)
        
        distance = np.sum(Weighting[comp])
        
    else:
        distance = np.sum(comp)
    
    return distance