class Quasiparticle:
    '''
    Container object for a generalized quasiparticle with arbitrary mass and charge.
    Tracks position and velocity.
    '''
    def __init__(self, q, m, pos0, vel0):
        self.q = q
        self.m = m
        self.pos = []
        self.vel = []
        self.pos.append(pos0)
        self.vel.append(vel0)

    def addPos(self, pos):
        self.pos.append(pos)

    def addVel(self, vel):
        self.vel.append(vel)

    def reset(self):
        p0 = self.pos[0]
        v0 = self.vel[0]
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

def initializeChargeCloud(q, m, N, R, R0):
    '''
    Create a list of Quasiparticle objects along a path smeared with a gaussian distribution
    :param q: charge in Coulomb of individual quasiparticle
    :param m: mass in kg of individual quasiparticle
    :param N: number of quasiparticles to create
    :param R: Radius of gaussian smoothing (sigma)
    :param R0: either (DIMENSIONS) array or (N, DIMENSIONS) array describing initial positions
    '''
    pos = R0 + R*np.random.normal(size=(N, DIMENSIONS))
    velS = maxwell.rvs(size=N, scale=scale)
    velDir = unitSphereVector(N)
    
    cc = []
    for i in range(N):
        cc.append(Quasiparticle(q, m, pos[i], velS[i]*velDir[i]))
    return cc
