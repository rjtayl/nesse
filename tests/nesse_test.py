# -*- coding: utf-8 -*-
import os
import sys
# sys.path.append(os.getcwd()+"/src/")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import nesse
import numpy as np

'''
A test of the basic NESSE stack. Basic meaning we are not using any "advanced" features like parallelization or
plotting.
TODO: check multiple ways of doing things e.g. not calling setWeightingField before calculateInducedCurrent
'''

def test_integration():
    ID = "4"

    events_filename = "config/Events/e-_800keV_0inc.root"
    Events = nesse.eventsFromG4root(events_filename)[:1]

    EF_filename = "config/Fields/4e10/NessieEF_Base4e7Linear0-150.0V.hf"
    WP_filename = "config/Fields/NessieWP_4e7Linear0-150V_grid.hf"

    Efield=nesse.fieldFromH5(EF_filename, rotate90=True)  
    weightingPotential = nesse.potentialFromH5(WP_filename, rotate90=True)

    sim = nesse.Simulation("Example_sim", 125, Efield, _weightingPotential=weightingPotential, contacts=1)

    spiceFile= "g:/nesse/config/Spice/spice_step_New_1ns.csv"
    sim.setElectronicResponse(spiceFile)

    ef_bounds = [[axis[0],axis[-1]] for axis in Efield.grid]
    bounds = np.stack((ef_bounds[0],ef_bounds[1],[0,0.002]))
    sim.setBounds(bounds)

    sim.simulate(Events, ds=1e-6, diffusion=True, dt=1e-10, maxPairs=2, silence=True, parallel=False)

    sim.setWeightingField()

    sim.calculateInducedCurrent(Events, 1e-9, detailed=False, parallel=True)

    sim.calculateElectronicResponse(Events, parallel=True)

    return None
    
if __name__ == "__main__":
    test_integration()  