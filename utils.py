import numpy as np
from scipy import special


#utils for modes

def hermite(X, N):              #hermite polynomial
    HER = special.hermite(N)
    sn = HER(X)
    return sn

def laguerre(X, L, P):          #laguerre polynomial
    LAG = special.genlaguerre(P, L)
    sn = LAG(X)
    return sn



#utils for Beam class parameters calculation

