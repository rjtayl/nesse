from .constants import *
from numba import jit
import numpy as np
from math import *

#Note everything is in meters. This is not standard in the literature on mobility.

beta_el = 1 # beta coefficient in electron mobility parametrization according to Knoll
beta_h = 2
vstar_el = 2.4e5 # m/s
Theta_el = 600 # K
C_el = 0.8
# TODO: Hole values are just sort of copy pasted, needs to be actually checked
vstar_h = 1e5 # m/s
Theta_h = 600 # K
C_h = 0.8
D0_el = 0.0036 # m^2/s
D0_h = 0.0012 # m^2/s

@jit(nopython=True)
def mu0_el(T):
    return 0.1440 * (T/300)**-2.01 # m^2/V/s

@jit(nopython=True)
def mu0_h(T):
    return 0.0450 * (T/300)**-2.01 # m^2/V/s

# Temperature-dependent saturation velocity
@jit(nopython=True)
def vs_el(T): 
    return vstar_el/(1+C_el*exp(T/Theta_el))

@jit(nopython=True)
def vs_h(T):
    return vstar_h/(1+C_h*exp(T/Theta_h))

@jit(nopython=True)
def diffusion_electron(T):
    return D0_el*(T/300)**-1.01

@jit(nopython=True)
def diffusion_hole(T):
    return D0_h*(T/300)**-1.01

def mobility(mu, E, beta, vs):
    return mu/(1+(mu*E/vs)**(1/beta))**beta

# Impurity density dependent mobility parametrization according to Klaassen
mu_max = mu0_el(300)
mu_min = 0.00685 # m^2/V/s
Nref1 = 9.2e10 # m^-3
Nref2 = 3.41e14 # m^-3
alpha1 = 0.711

@jit(nopython=True, fastmath=True)
def mu_I_N(T): 
    return mu_max**2/(mu_max-mu_min)*(T/300)**(3*alpha1-1.5)

@jit(nopython=True, fastmath=True)
def mu_I_c(T):
    return mu_min*mu_max/(mu_max-mu_min)*(300/T)**0.5

@jit(nopython=True, fastmath=True)
def mu_I(T, NI):
    return mu_I_N(T)*(Nref1/NI)**alpha1 + mu_I_c(T)

def generalized_mobility_el(T, NI, E):
    mu = 1/(1/mu0_el(T)+1/mu_I(T, NI))
    vsat = vs_el(T)
    return mu/(1+(mu*E/vsat)**(1/beta_el))**beta_el

@jit(nopython=True)
def generalized_mobility_h(T, NI, E):
    mu = 1/(1/mu0_h(T)+1/mu_I(T, NI))
    vsat = vs_h(T)
    return mu/(1+(mu*E/vsat)**(1/beta_h))**beta_h

def generalized_diffusion_el(T, NI, E):
    return mobility(T, NI, E)*T*kB
