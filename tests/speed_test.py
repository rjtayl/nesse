# -*- coding: utf-8 -*-
import os, sys, cProfile
import numpy as np
sys.path.append(os.getcwd()+"/src/")
import nesse

#@profile
def main(threads):
    ID = "4"

    print("Loading Data")
    events_filename = "config/Events/e-_800keV_0inc.root"
    Events = nesse.eventsFromG4root(events_filename)[:1]

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

    sim.setThreadNumber(threads)

    print("Drifting Quasiparticles")
    sim.simulate(Events, ds=1e-6, diffusion=True, dt=1e-10, maxPairs=100, silence=True, parallel=True)

    sim.setWeightingField()

    print("Calculating induced current")
    sim.calculateInducedCurrent(Events, 1e-9, detailed=False, parallel=True)

    print("Calculating electronic response")
    sim.calculateElectronicResponse(Events)

    print("Saving Events")
    nesse.saveEventsNabPy(Events, "tempEvents")

    return None

if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()

    # nthreads_available = len(os.sched_getaffinity(0)) #only works on UNIX
    nthreads_available = os.cpu_count()
    print(f'Threads: {nthreads_available=}')

    main(6) # On my machine (Windows AMD 5700x3D) this runs the fastest for 10 quasiparticles per deposition.
            # This is probably a balance of number of quasiparticles per event and overhead of managing extra threads. 
    pr.disable()

    # Dump results:
    # - for binary dump
    pr.dump_stats('output.prof')
    # - for text dump
    with open( 'output.txt', 'w') as output_file:
        sys.stdout = output_file
        pr.print_stats( sort='time' )
        sys.stdout = sys.__stdout__