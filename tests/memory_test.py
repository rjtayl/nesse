import os
import sys
sys.path.append(os.getcwd()+"/src/")
import nesse
import numpy as np

import tracemalloc

def main():
    ID = "4"
    print(tracemalloc.get_traced_memory())

    events_filename = "config/Events/e-_800keV_0inc.root"
    Events = nesse.eventsFromG4root(events_filename)[:5]

    print("Events:")
    print(tracemalloc.get_traced_memory())

    EF_filename = "config/Fields/NessieEF_Base4e7Linear0-150.0V.hf"
    WP_filename = "config/Fields/NessieWP_4e7Linear0-150V_grid.hf"

    Efield=nesse.fieldFromH5(EF_filename, rotate90=True)  
    print("EField:")
    print(tracemalloc.get_traced_memory())


    weightingPotential = nesse.potentialFromH5(WP_filename, rotate90=True)
    print("WP:")
    print(tracemalloc.get_traced_memory())

    sim = nesse.Simulation("Example_sim", 125, Efield, _weightingPotential=weightingPotential, contacts=2)

    spiceFile= "g:/nesse/config/Spice/spice_step_New_1ns.csv"
    sim.setElectronicResponse(spiceFile)

    sim.setIDP(lambda x, y, z : float(ID) * 1e16)

    ef_bounds = [[axis[0],axis[-1]] for axis in Efield.grid]
    bounds = np.stack((ef_bounds[0],ef_bounds[1],[0,0.002]))
    sim.setBounds(bounds)

    print("Sim:")
    print(tracemalloc.get_traced_memory())

    sim.simulate(Events, ds=1e-6, diffusion=True, dt=1e-10, maxPairs=1, silence=True)

    print("Simulated:")
    print(tracemalloc.get_traced_memory())

    # sim.setWeightingField()

    # print("Weighting Field:")
    # print(tracemalloc.get_traced_memory())

    sim.calculateInducedCurrent(Events, 1e-9)
    print("Induced Current:")
    print(tracemalloc.get_traced_memory())

    sim.calculateElectronicResponse(Events)

    return None

if __name__ == "__main__":
    tracemalloc.start()
    main()
    print(tracemalloc.get_traced_memory())
    tracemalloc.stop()