import numpy as np
from scipy.stats import maxwell

class Quasiparticle:
    '''
    Container object for a generalized quasiparticle with arbitrary mass and charge.
    Tracks position and velocity.
    '''
    def __init__(self, q, m, t0, pos0, vel0):
        self.q = q
        self.m = m
        self.pos = [pos0,]
        self.vel = [vel0,]
        self.time = [t0,]
        self.alive = True

    def addTime(self, t):
        self.time.append(t)

    def addPos(self, pos):
        self.pos.append(pos)

    def addVel(self, vel):
        self.vel.append(vel)

    def reset(self):
        t0 = self.time[0]
        p0 = self.pos[0]
        v0 = self.vel[0]
        self.t0 = [t0, ]
        self.pos = [p0, ]
        self.vel = [v0, ]

def unitSphereVector(N):
    '''
    Sample N points uniformly on a sphere of radius 1
    '''
    phi = 2*np.pi*np.random.uniform(size=N)
    z = 1-2*np.random.uniform(size=N)
    theta = np.arccos(z)
    return np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), z]).T

def initializeChargeCloud(q, m, N, t0, R, R0):
    '''
    Create a list of Quasiparticle objects along a path smeared with a gaussian distribution
    :param q: charge in Coulomb of individual quasiparticle
    :param m: mass in kg of individual quasiparticle
    :param N: number of quasiparticles to create
    :param R: Radius of gaussian smoothing (sigma)
    :param R0: either (DIMENSIONS) array or (N, DIMENSIONS) array describing initial positions
    '''
    pos = R0 + R*np.random.normal(size=(N, 3))
    #velS = maxwell.rvs(size=N, scale=scale)
    #TODO fix scale for Maxwell
    velS = np.zeros(N)
    velDir = unitSphereVector(N)
    
    cc = []
    for i in range(N):
        cc.append(Quasiparticle(q, m, t0, pos[i], velS[i]*velDir[i]))
    return cc
