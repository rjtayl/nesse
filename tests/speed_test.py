# -*- coding: utf-8 -*-
import os, sys, cProfile
import numpy as np
sys.path.append(os.getcwd()+"/src/")
import nesse

#@profile
def main():
    ID = "4"

    print("Loading Data")
    events_filename = "config/Events/e-_800keV_0inc.root"
    Events = nesse.eventsFromG4root(events_filename)[:10]

    EF_filename = "config/Fields/4e10/NessieEF_Base4e7Linear0-150.0V.hf"
    WP_filename = "config/Fields/NessieWP_4e7Linear0-150V_grid.hf"
    # WP_filename = "config/Fields/NessieWP_7PixelWP-finemesh.hf"

    Efield=nesse.fieldFromH5(EF_filename, rotate90=True)  
    weightingPotential = nesse.potentialFromH5(WP_filename, rotate90=True)

    sim = nesse.Simulation("Example_sim", 125, Efield, _weightingPotential=weightingPotential, contacts=1)

    spiceFile= "g:/nesse/config/Spice/spice_step_New_1ns.csv"
    sim.setElectronicResponse(spiceFile)

    # sim.setIDP(lambda x, y, z : float(ID) * 1e16)

    ef_bounds = [[axis[0],axis[-1]] for axis in Efield.grid]
    bounds = np.stack((ef_bounds[0],ef_bounds[1],[0,0.002]))
    sim.setBounds(bounds)

    # nthreads_available = len(os.sched_getaffinity(0)) #only works on UNIX
    nthreads_available = os.cpu_count()
    print(f'Threads: {nthreads_available=}')
    sim.setThreadNumber(nthreads_available)

    print("Drifting Quasiparticles")
    sim.simulate(Events, ds=1e-6, diffusion=True, dt=1e-10, maxPairs=10, silence=True, parallel=True)

    sim.setWeightingField()

    print("Calculating induced current")
    sim.calculateInducedCurrent(Events, 1e-9, detailed=False)

    print("Calculating electronic response")
    sim.calculateElectronicResponse(Events)

    print("Saving Events")
    nesse.saveEventsNabPy(Events, "tempEvents")

    return None

if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()
    main()
    pr.disable()

    # Dump results:
    # - for binary dump
    pr.dump_stats('output.prof')
    # - for text dump
    with open( 'output.txt', 'w') as output_file:
        sys.stdout = output_file
        pr.print_stats( sort='time' )
        sys.stdout = sys.__stdout__