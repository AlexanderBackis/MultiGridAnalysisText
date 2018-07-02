import numpy as np
from scipy.optimize import curve_fit
from scipy.special import factorial

def fit_poission(x,y):
    norm = np.sum(y)
    y = y/norm
    param = curve_fit(poisson, x, y)
    return param
    
def poisson(k, lamb):
    return (lamb**k/factorial(k)) * np.exp(-lamb)


