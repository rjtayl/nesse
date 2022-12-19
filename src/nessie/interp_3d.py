import numpy as np
import pyximport
pyximport.install(setup_args={'include_dirs': [np.get_include()]})

from numba import jit

from .interp import _interp3D

@jit(nopython=True)
def find_first(item, vec):
    for i, val in enumerate(vec):
        #print(item, vec[i])
        if item < val:
            return i
    return -1

class Interp3D(object):
    def __init__(self, v, x, y, z):
        self.v = v
        self.min_x, self.max_x = x[0], x[-1]
        self.min_y, self.max_y = y[0], y[-1]
        self.min_z, self.max_z = z[0], z[-1]
        self.delta_x = (self.max_x - self.min_x)/(x.shape[0]-1)
        self.delta_y = (self.max_y - self.min_y)/(y.shape[0]-1)
        self.delta_z = (self.max_z - self.min_z)/(z.shape[0]-1)

    def __call__(self, t):
        X,Y,Z = self.v.shape[0], self.v.shape[1], self.v.shape[2]

        x = (t[0]-self.min_x)/self.delta_x
        y = (t[1]-self.min_y)/self.delta_y
        z = (t[2]-self.min_z)/self.delta_z


        return _interp3D(self.v, x, y, z, X, Y, Z)
        
        
class Interp3D(object):
    def __init__(self, v, x, y, z):
        self.v = v
        self.x = x
        self.y = y
        self.z = z
        self.delta_x = np.diff(x)
        self.delta_y = np.diff(y)
        self.delta_z = np.diff(z)

    def __call__(self, t):
        X,Y,Z = self.v.shape[0], self.v.shape[1], self.v.shape[2]
        
        #print(t)

        #i = np.where(self.x>t[0])[0][0]-1
        #j = np.where(self.y>t[1])[0][0]-1
        #k = np.where(self.z>t[2])[0][0]-1
        
        i = find_first(t[0], self.x)-1
        j = find_first(t[1], self.y)-1
        k = find_first(t[2], self.z)-1
        
        
        #print(i,j,k)
        #print(self.x[i], self.x[j], self.x[k])
    
        l = i + (t[0]-self.x[i])/self.delta_x[i]
        m = j + (t[1]-self.y[j])/self.delta_y[j]
        n = k + (t[2]-self.z[k])/self.delta_z[k]
        
        #print(l,m,n)

        return _interp3D(self.v, l, m, n, X, Y, Z)
