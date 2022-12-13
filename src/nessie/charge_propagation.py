from .constants import *
from .mobility import *
import numpy as np

def CoulombForce(q1, q2, pos1, pos2, safety = 7e-9):
    r2 = max(safety**2, sum((pos1-pos2)**2))
    Fs = -kE*q1*q2/r2/epsR
    return Fs*(pos2-pos1)/r2**0.5

def getDriftStep(angle, d, dt):
    #print(d, dt)
    sigma1, sigma2, sigma3 = np.random.standard_normal(3)*(2*d*dt)**0.5
    #dxd = np.cos(angle)*sigma1 - np.sin(angle)*sigma2
    #dyd = np.sin(angle)*sigma1 + np.cos(angle)*sigma2
    return sigma1, sigma2, sigma3

def getEffectiveCoulombField(objects, dt):
    acc = np.zeros((len(objects), DIMENSIONS))
    for i in range(len(objects)):
        Q = objects[i].q
        M = objects[i].m
        a = np.zeros(DIMENSIONS)
        for j in range(len(objects)):
            if i != j:
                a+= calcForce(Q, objects[j].q, objects[i].pos[-1], objects[j].pos[-1])/M
        acc[i] = a

def updateQuasiparticles(objects, dt, extField = np.zeros(DIMENSIONS), temp=300):
    for i in range(len(objects)):
        Eeff = extField + acc[i]*objects[i].m/objects[i].q/100 # factor 100 is to convert is to V/cm instead of V/m
        mu = mobility(mu0_el(temp) if objects[i].q < 0 else mu0_h(temp), np.linalg.norm(Eeff), beta, vs)
        dv = mu*Eeff/100*objects[i].q/abs(objects[i].q) # factor 100 is to convert from cm/s to m/s
        dv += ((D_el(temp) if objects[i].q < 0 else D_h(temp))/dt)**0.5/100*np.random.normal(size=3)
        objects[i].addVel(dv)
        objects[i].addPos(objects[i].pos[-1]+objects[i].vel[-1]*dt)

def propagateCarrier(x0, y0, z0, eps, Ex_i, Ey_i, Ez_i, E_i, bounds, T, tauTrap = lambda x,y,z: 1e9, d=None, NI=lambda x, y,z: 1e10, diffusion=False, electron=True, interp3d=False):
    
    x = [x0, ]
    y = [y0, ]
    z = [z0, ]
    t = [0, ]
    diffRatio = [0, ]
    while ((x[-1] >= bounds[0][0]) & (x[-1] <= bounds[0][1]) & (y[-1] >= bounds[1][0]) & (y[-1] <= bounds[1][1]) 
            & (z[-1] >= bounds[2][0]) & (z[-1] <= bounds[2][1])):
        #print(len(t), len(t)<stepLimit)
        #print(x[-1], y[-1], z[-1])
        
        if interp3d:
            E = E_i([x[-1], y[-1], z[-1]])
            Ex = Ex_i([x[-1], y[-1], z[-1]])
            Ey = Ey_i([x[-1], y[-1], z[-1]])
            Ez = Ez_i([x[-1], y[-1], z[-1]])
        
        else:
            E = E_i([x[-1], y[-1], z[-1]])[0]
            Ex = Ex_i([x[-1], y[-1], z[-1]])[0]
            Ey = Ey_i([x[-1], y[-1], z[-1]])[0]
            Ez = Ez_i([x[-1], y[-1], z[-1]])[0]
        
        
        if electron:
        	mu = generalized_mobility_el(T, NI(x[-1], y[-1], z[-1]), E)
        else:
        	mu = mu0_h(T)
        	
        if d == None:
            #d_temp = D(T, NI(x[-1], y[-1], z[-1]), E)
            d_temp = diffusion_electron(T) if electron else diffusion_hole(T)
        else:
            d_temp = d


        dx = 0
        dy = 0
        dz = 0
        
        if abs(E) > 0:
            dt = min(min(abs(eps/mu/E), tauTrap(x[-1], y[-1], z[-1])/1e5),(bounds[2][1]-z[-1])/mu/np.abs(Ez)+1e-10) #Time step needs to be significantly smaller than trapping time for Poisson process to make sense 
            
            sign = -1 if electron else 1
            dx = sign*dt*mu*Ex
            dy = sign*dt*mu*Ey
            dz = sign*dt*mu*Ez
            dr = np.sqrt(dx**2+dy**2+dz**2)
            
            #print(dt, dz)
            angle = 0
            if (dx != 0):
                angle = np.arctan(dy/dx)
            phi = np.arccos(dz/dr)
        else:
            dt = tauTrap(x[-1], y[-1], z[-1])/1e5
            angle = 0
        
        #Check if particle gets trapped
        #p = dt/tauTrap(x[-1], y[-1], z[-1])
        #r = np.random.random()
        #if r < p:
        #    return x, y, z, t
        
        
        dxd = 0
        dyd = 0
        dzd = 0
       
        if (diffusion and d_temp != 0):
                dxd, dyd, dzd = getDriftStep(angle, d_temp, dt)
        
        #print("diffusion", dxd, dyd, dzd, dt)
        
        x.append(x[-1] + dx+dxd)
        y.append(y[-1] + dy+dyd)
        z.append(z[-1] + dz+dzd)
        t.append(t[-1] + dt)
        
        #print(x[-1], y[-1], t[-1])
        
    return x, y, z, t

if __name__ == "__main__":
    x, y, z, t = propagateCarrier(0, 0, 0, 1e-4, lambda a : [0,], lambda a : [0,], lambda a: [-750e2,], lambda a : [-750e2,], [[-1, 1], [-1, 1], [0, 0.002]], 130, diffusion=True)
