# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.getcwd()+"/src/")
import nessie
import numpy as np
import matplotlib.pyplot as plt
import cProfile
import pstats
from pstats import SortKey

#@profile
def main():
    #import SSD fields
    EF_filename = "config/Fields/NessieEF_4e7Linear0-150V_grid.hf"

    Efield=nessie.eFieldFromH5(EF_filename)  

    eFieldx_interp, eFieldy_interp, eFieldz_interp, eFieldMag_interp = Efield.interpolate(True)
    
    #Ef = [eFieldz_interp([0,0,z]) for z in np.linspace(-0.001,0.002,10000)]
    
    zs = list(np.linspace(-0.001,0.002,100000))
    cProfile.runctx('[eFieldz_interp([0,0,z]) for z in zs]', {'eFieldz_interp':eFieldz_interp, 'zs':zs},{}, 'interp_stats')
    
    p = pstats.Stats('interp_stats')
    p.sort_stats(SortKey.FILENAME).print_stats('interp_3d.py')
    p.sort_stats(SortKey.FILENAME).print_stats('_interp3D')

if __name__ == "__main__":
    #cProfile.run('main()')
    main()  
