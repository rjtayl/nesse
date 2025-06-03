import os
import sys
sys.path.append(os.getcwd()+"/src/")
import nesse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator 
import time


def main():
    EF_filename = "config/Fields/NessieEF_4e7Linear0-150V_grid.hf"
    WP_filename = "config/Fields/NessieWP_4e7Linear0-150V_grid.hf"

    Efield=nesse.eFieldFromH5(EF_filename)  
    weightingPotential = nesse.weightingPotentialFromH5(WP_filename)

    x = weightingPotential.grid[0].astype("double")
    y = weightingPotential.grid[1].astype("double")
    z = weightingPotential.grid[2].astype("double")
    
    v = weightingPotential.data.astype("double",order="C")
    
    interp = nesse.Interp3D(v,x,y,z)
    interp_si = RegularGridInterpolator((x,y,z),v)
    
    zs = np.linspace(-0.001,0.003, 1000)
    
    t0 = time.time()
    wp = [interp([0,0,z]) for z in zs]
    t1 = time.time() 
    wp_si = [interp_si([0,0,z])[0] for z in zs]
    t2 = time.time()
    print(t1-t0, t2-t1)
    plt.plot(zs,wp)
    plt.plot(zs,wp_si)
    plt.show()
    
    
if __name__ == "__main__":
    main()
