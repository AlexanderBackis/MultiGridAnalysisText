import numpy as np
from scipy.optimize import curve_fit
from scipy.special import factorial
import matplotlib.pyplot as plt

def fit_poisson(x,y):
    norm = np.sum(y)
    sigma = np.sqrt(y)
    y = y/norm
    sigma = sigma/norm
    param = curve_fit(poisson, x, y, p0=None, sigma=sigma)
    return param
    
def poisson(k, lamb):
    return (lamb**k/factorial(k)) * np.exp(-lamb)





