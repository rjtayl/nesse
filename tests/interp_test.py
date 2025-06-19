# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.getcwd()+"/src/")
import nesse
import numpy as np
import matplotlib.pyplot as plt
import cProfile
import pstats
from pstats import SortKey
import time

#@profile
def main():
    #import SSD fields
    EF_filename = "config/Fields/4e10/NessieEF_Base4e7Linear0-150.0V.hf"

    Efield=nesse.fieldFromH5(EF_filename, rotate90=True)  

    eFieldx_interp, eFieldy_interp, eFieldz_interp, eFieldMag_interp = Efield.interpolate(True)
    
    #Ef = [eFieldz_interp([0,0,z]) for z in np.linspace(-0.001,0.002,10000)]
    
    zs = list(np.linspace(-0.001,0.002,100000))
    coords = [[0,0,z] for z in zs]
    cProfile.runctx('[eFieldz_interp([0,0,z]) for z in zs]', {'eFieldz_interp':eFieldz_interp, 'zs':zs},{}, 'interp_stats')

    p1 = pstats.Stats('interp_stats')
    p1.sort_stats(SortKey.FILENAME).print_stats('interp_3d.py')
    p1.sort_stats(SortKey.FILENAME).print_stats('_interp3D')
    
    #Note: cProfile doesnt appear to play nice with jit functions so its reporting a much longer run time than real
    cProfile.runctx('eFieldz_interp(coords)', {'eFieldz_interp':eFieldz_interp, 'coords':coords},{}, 'interp_stats_vector')
    
    p = pstats.Stats('interp_stats_vector')
    p.sort_stats(SortKey.FILENAME).print_stats('interp_3d.py')
    p.sort_stats(SortKey.FILENAME).print_stats('_interp3D')

    t0 = time.time()
    efields_old = [eFieldz_interp([0,0,z]) for z in zs]
    t1 = time.time()
    efields_new = eFieldz_interp(coords)
    t2 = time.time()

    print(f"Old time: {t1-t0} \n New time: {t2-t1}")

    diff = efields_new-efields_old
    print(np.average(diff))
    
if __name__ == "__main__":
    #cProfile.run('main()')
    main()  
