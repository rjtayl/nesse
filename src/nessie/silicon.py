from constants import *

def Egap(T):
    a = 4.9e-4
    b = 655 # K
    return E0gap - a*T*T/(T+b)

def ephCanali(T):
    return 2.15*Egap(T)+1.21

def ephBestFit(T):
    return 4.26*Egap(T)-1.15