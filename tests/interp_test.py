# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.getcwd()+"/src/")
import nesse
import numpy as np

def test_interp():
    potential_func = lambda x,y,z: x + y + 2*z
    bounds = [[-1,1],[-1,1],[0,2]]

    ds = 0.1
    
    potential=nesse.analytical_potential("test", potential_func, bounds, ds, ds, ds)

    x = potential.grid[0].astype(np.float32)
    y = potential.grid[1].astype(np.float32)
    z = potential.grid[2].astype(np.float32)
    
    v = potential.data.astype(np.float32,order="C")
    
    interp = nesse.Interp3D(v,x,y,z)
    
    #check basic linear interpolation
    positions = np.array([[0,0,0], [0.5,0.5,1], [0.5,0.5,0.15]], dtype=np.float32)

    expected = [np.round(potential_func(*pos), 5) for pos in positions]
    interped = [np.round(interp(pos), 5) for pos in positions]

    assert expected == interped, f"Interpolation does not match analytical results: {expected}, {interped}"

    # check out of bounds
    positions = np.array([[-1.05,0,0],[-1,-1,-2], [1,1,2], [2,0,0]], dtype=np.float32)
    expected = [0 for pos in positions]
    interped = [np.round(interp(pos), 5) for pos in positions]

    t=positions[-1]
    i, j, k = interp.get_ijk(t)
    l, m, n = interp.get_lmn(t, i, j, k)

    assert expected == interped, f"Interpolation returns non-zero result out of bounds: {interped}"

    #check vectorization gets same results

    positions = np.array([[0,0,0], [0.5,0.5,1], [0.5,0.5,0.15], [-1.05,0,0],[-1,-1,-2], [1,1,2], [2,0,0]], dtype=np.float32)

    interped = np.array([np.round(interp(pos), 5) for pos in positions],dtype=np.float32)

    interped_vector =np.array([np.round(val, 5) for val in interp(positions)],dtype=np.float32)

    assert np.all(interped_vector == interped), f"Vectorized interpolation not working: {interped}, {interped_vector}"

if __name__ == "__main__":
    test_interp()  