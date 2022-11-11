from .constants import *

mu0_el = lambda T: 1440 * (T/300)**-2.01 # cm^2/V/s
mu0_h = lambda T: 450 * (T/300)**-2.01 # cm^2/V/s
beta_el = 1 # beta coefficient in electron mobility parametrization according to Knoll
vstar = 2.4e7 # cm/s
Theta = 600 # K
C = 0.8
vs_el = lambda T: vstar/(1+C*np.exp(T/Theta)) # Temperature-dependent saturation velocity

D0_el = 36 # cm^2/s
D_el = lambda T: D0_el*(T/300)**-1.01
D0_h = 12 # cm^2/s
D_h = lambda T: D0_h*(T/300)**-1.01

def mobility(mu, E, beta, vs):
    return mu/(1+(mu*E/vs)**(1/beta))**beta

# Impurity density dependent mobility parametrization according to Klaassen
mu_max = mu0_el(300)
mu_min = 68.5 # cm^2/V/s
Nref1 = 9.2e16 # cm^-3
Nref2 = 3.41e20 # cm^-3
alpha1 = 0.711

mu_I_N = lambda T: mu_max**2/(mu_max-mu_min)*(T/300)**(3*alpha1-1.5)
mu_I_c = lambda T: mu_min*mu_max/(mu_max-mu_min)*(300/T)**0.5

mu_I = lambda T, NI: mu_I_N(T)*(Nref1/NI)**alpha1 + mu_I_c(T)

def generalized_mobility_el(T, NI, E):
    mu = 1/(1/mu0(T)+1/(mu_I(T, NI)))
    vsat = vs_el(T)
    return mu/(1+(mu*E/vsat)**(1/beta_el))**beta_el

def generalized_diffusion_el(T, NI, E):
    return mobility(T, NI, E)*T*kB
