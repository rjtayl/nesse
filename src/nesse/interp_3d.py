import numpy as np
import pyximport
pyximport.install(setup_args={'include_dirs': [np.get_include()]})

from numba import jit, guvectorize, int32, int64, float32, float64

from .interp import _interp3D # type: ignore

@jit(nopython=True)
def find_first(item, vec):
    for i, val in enumerate(vec):
        if item < val:
            return i
    return -1

@guvectorize([(float64, float64[:], int64[:])], '(),(n)->()', nopython=True)
def find_first_vector(item, vec, out):
    out[0] = -1
    for i in range(vec.shape[0]):
        if item < vec[i]:
            out[0] = i
            break  
 

@guvectorize([(float64[:], float64[:], float64[:], float64[:], int64[:])], 
             '(d),(l),(m),(n)->(d)', nopython=True)
def get_ijk_vector(t, x, y, z, out):
    axes = [x, y, z]
    for d in range(3):
        out[d] = 0
        for i in range(axes[d].shape[0]):
            if t[d] < axes[d][i]:
                out[d] = i - 1
                break
    

class Interp3D_old(object):
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
    
    def get_lmn_vector(self, t, i, j, k):
        #tried converting to jit, didn't speed up anything. 
        # Vectorized version: t is (N, 3), i, j, k are arrays of length N
        l = i + (t[:, 0] - self.x[i]) / self.delta_x[i]
        m = j + (t[:, 1] - self.y[j]) / self.delta_y[j]
        n = k + (t[:, 2] - self.z[k]) / self.delta_z[k]
        return l, m, n

    def __call__(self, t):
        # Allow t to be a single coordinate (shape (3,)) or an array of coordinates (shape (N, 3))
        t = np.asarray(t)
        if t.ndim == 1:
            # Single coordinate
            X, Y, Z = self.v.shape
            i, j, k = self.get_ijk(t)
            l, m, n = self.get_lmn(t, i, j, k)
            return _interp3D(self.v, l, m, n, X, Y, Z)
        elif t.ndim == 2 and t.shape[1] == 3:
            # Array of coordinates
            X, Y, Z = self.v.shape

            ijks= get_ijk_vector(t, self.x, self.y, self.z)
            i = ijks[:, 0]
            j = ijks[:, 1]
            k = ijks[:, 2]

            l, m, n = self.get_lmn_vector(t, i, j, k)

            #TODO: vectorize _interp3D
            results = np.empty(t.shape[0], dtype=self.v.dtype)
            for idx, coord in enumerate(t):
                results[idx] = _interp3D(self.v, l[idx], m[idx], n[idx], X, Y, Z)
            return results
        else:
            raise ValueError("Input t must be shape (3,) or (N, 3)")

