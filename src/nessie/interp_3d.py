import numpy as np
import pyximport
pyximport.install(setup_args={'include_dirs': [np.get_include()]})

from numba import jit

from .interp import _interp3D

@jit(nopython=True)
def find_first(item, vec):
    for i, val in enumerate(vec):
        if item < val:
            return i
    return -1

@jit(nopython=True)
def bisect_left(a, x):
    hi = len(a)
    lo = 0
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo
 
@jit(nopython=True)
def get_ijk(t, x, y, z):
    i_s = [0,0,0]
    axes = [x,y,z]
    for d in range(3):
        for i, val in enumerate(axes[d]):
            if t[d] < val:
                i_s[d] = i -1
                
    return i_s
    

class Interp3D(object):
    '''
    Grid interpolator for 3-dimensional regular rectangular grid. 
    This is an old version, see function below.
    '''
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
    '''
    Grid interpolator for 3-dimensional rectangular grid. 
    Note that this is changed from above to work for non-regular grids.
    get_ijk and get_lmn are seperate functions mainly for testing purposes.
    '''
    def __init__(self, v, x, y, z):
        self.v = v
        self.x = x
        self.y = y
        self.z = z
        self.delta_x = np.diff(x)
        self.delta_y = np.diff(y)
        self.delta_z = np.diff(z)
        
    def get_ijk(self, t):
        return find_first(t[0], self.x)-1, find_first(t[1], self.y)-1, find_first(t[2], self.z)-1

    def get_lmn(self, t, i, j, k):
        return i + (t[0]-self.x[i])/self.delta_x[i], j + (t[1]-self.y[j])/self.delta_y[j], k + (t[2]-self.z[k])/self.delta_z[k]

    def __call__(self, t):
        X,Y,Z = self.v.shape[0], self.v.shape[1], self.v.shape[2]

        #i = np.where(self.x>t[0])[0][0]-1
        #j = np.where(self.y>t[1])[0][0]-1
        #k = np.where(self.z>t[2])[0][0]-1

        i,j,k = self.get_ijk(t)
        
        l,m,n = self.get_lmn(t, i, j, k)

        return _interp3D(self.v, l, m, n, X, Y, Z)
