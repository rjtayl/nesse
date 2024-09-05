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
Nref1 = 9.2e22 # m^-3
Nref2 = 3.41e26 # m^-3
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

@jit(nopython=True)
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

#mobility determined from fits to canali data
#electrons
mu_0_e = 1635*1e-4
v_s_e = 1.04e7*1e-2
c_e = 0.9438

theta_mu_e = 2.462
theta_v_s_e = 0.638
theta_c_e = -0.7185

#holes
mu_0_h = 508*1e-4
v_s_h = 9.98e6*1e-2
c_h = 1.077

theta_mu_h = 2.26
theta_v_s_h = 0.09
theta_c_h = -0.24

def canali_mobility_e(T, NI, E):
    mu = mu_0_e * (300/T)**theta_mu_e
    v_s = v_s_e * (300/T)**theta_v_s_e
    c = c_e * (300/T)**theta_c_e
    return mu/(1+(mu*E/v_s)**c)**(1/c)

def canali_mobility_h(T, NI, E):
    mu = mu_0_h * (300/T)**theta_mu_h
    v_s = v_s_h * (300/T)**theta_v_s_h
    c = c_h * (300/T)**theta_c_h
    return mu/(1+(mu*E/v_s)**c)**(1/c)


mu_0_e_nab = (1635*1e-4) * (1 - 0.07)
def Nab_canali_mobility_e(T, NI, E):
    mu = mu_0_e_nab * (300/T)**theta_mu_e
    v_s = v_s_e * (300/T)**theta_v_s_e
    c = c_e * (300/T)**theta_c_e
    return mu/(1+(mu*E/v_s)**c)**(1/c)

def Nab_canali_mobility_h(T, NI, E):
    mu = mu_0_h * (300/T)**theta_mu_h
    v_s = v_s_h * (300/T)**theta_v_s_h
    c = c_h * (300/T)**theta_c_h
    return mu/(1+(mu*E/v_s)**c)**(1/c)

#RJ fit to electrons
@jit(nopython=True)
def mu0_el_RJ(T):
    return 0.1521 * (T/300)**-2.01 # m^2/V/s

#RJ max fit  to electrons
@jit(nopython=True)
def mu0_el_RJ_max(T):
    return 0.1721 * (T/300)**-2.01 # m^2/V/s

@jit(nopython=True)
def generalized_mobility_el_RJ(T, NI, E):
    mu = 1/(1/mu0_el_RJ(T)+1/mu_I(T, NI))
    vsat = vs_el(T)
    return mu/(1+(mu*E/vsat)**(beta_el))**1/beta_el

# @jit(nopython=True)
# def mu0_h(T):
#     return 0.0450 * (T/300)**-2.247 # m^2/V/s


# @jit(nopython=True)
# def vs_h(T):
#     return mu0_h(T)/1.95e6

# @jit(nopython=True)
# def generalized_mobility_h(T, NI, E):
#     mu = 1/(1/mu0_h(T)+1/mu_I(T, NI))
#     vsat = vs_h(T)
#     return mu/(1+(mu*E/vsat)**(beta_h))**1/beta_h