from .constants import *
import numpy as np

#Note everything is in meters. This is not standard in the literature on mobility.

mu0_el = lambda T: 0.1440 * (T/300)**-2.01 # m^2/V/s
mu0_h = lambda T: 0.0450 * (T/300)**-2.01 # m^2/V/s
beta_el = 1 # beta coefficient in electron mobility parametrization according to Knoll
vstar = 2.4e5 # m/s
Theta = 600 # K
C = 0.8
vs_el = lambda T: vstar/(1+C*np.exp(T/Theta)) # Temperature-dependent saturation velocity

D0_el = 0.0036 # m^2/s
def diffusion_electron(T):
    return D0_el*(T/300)**-1.01

D0_h = 0.0012 # m^2/s
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

mu_I_N = lambda T: mu_max**2/(mu_max-mu_min)*(T/300)**(3*alpha1-1.5)
mu_I_c = lambda T: mu_min*mu_max/(mu_max-mu_min)*(300/T)**0.5

mu_I = lambda T, NI: mu_I_N(T)*(Nref1/NI)**alpha1 + mu_I_c(T)

def generalized_mobility_el(T, NI, E):
    mu = 1/(1/mu0_el(T)+1/(mu_I(T, NI)))
    vsat = vs_el(T)
    return mu/(1+(mu*E/vsat)**(1/beta_el))**beta_el

def generalized_diffusion_el(T, NI, E):
    return mobility(T, NI, E)*T*kB