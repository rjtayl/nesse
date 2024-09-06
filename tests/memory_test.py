import os
import sys
sys.path.append(os.getcwd()+"/src/")
import nesse
import numpy as np

import time

import tracemalloc

from sys import getsizeof, stderr
from itertools import chain
from collections import deque
try:
    from reprlib import repr
except ImportError:
    pass


def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)


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

    sim = nesse.Simulation("Example_sim", 125, Efield, _weightingPotential=weightingPotential, contacts=1)

    spiceFile= "g:/nesse/config/Spice/spice_step_New_1ns.csv"
    sim.setElectronicResponse(spiceFile)

    sim.setIDP(lambda x, y, z : float(ID) * 1e16)

    ef_bounds = [[axis[0],axis[-1]] for axis in Efield.grid]
    bounds = np.stack((ef_bounds[0],ef_bounds[1],[0,0.002]))
    sim.setBounds(bounds)

    print("Sim:")
    print(tracemalloc.get_traced_memory())

    sim.setThreadNumber(6)

    sim.simulate(Events, ds=1e-6, diffusion=True, dt=1e-10, maxPairs=1, silence=True, parallel=True)

    print("Simulated:")
    print(tracemalloc.get_traced_memory())

    # sim.setWeightingField()

    # print("Weighting Field:")
    # print(tracemalloc.get_traced_memory())

    print(total_size(Events))
    sim.calculateInducedCurrent(Events, 1e-9, detailed=True)
    print("Induced Current:")
    print(tracemalloc.get_traced_memory())
    print(total_size(Events))

    sim.calculateElectronicResponse(Events)

    return None

if __name__ == "__main__":
    tracemalloc.start()
    t0 = time.time
    main()
    t1 = time.time
    print(tracemalloc.get_traced_memory())
    tracemalloc.stop()

    print(t1-t0)