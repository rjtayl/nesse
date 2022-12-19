# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.getcwd()+"/src/")
import nessie
import numpy as np
import matplotlib.pyplot as plt
import cProfile

#@profile
def main():
    #import SSD fields
    EF_filename = "config/Fields/NessieEF_4e7Linear0-150V_grid.hf"

    Efield=nessie.eFieldFromH5(EF_filename)  

    eFieldx_interp, eFieldy_interp, eFieldz_interp, eFieldMag_interp = Efield.interpolate(True)
    
    Ef = [eFieldz_interp([0,0,z]) for z in np.linspace(-0.001,0.002,10000)]


if __name__ == "__main__":
    cProfile.run('main()')
    #main()  
