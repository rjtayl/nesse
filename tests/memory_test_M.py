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

        else:
            if not hasattr(o.__class__, '__slots__'):
                if hasattr(o, '__dict__'):
                    s+=sizeof(o.__dict__) # no __slots__ *usually* means a __dict__, but some special builtin classes (such as `type(None)`) have neither
                # else, `o` has no attributes at all, so sys.getsizeof() actually returned the correct value
            else:
                s+=sum(sizeof(getattr(o, x)) for x in o.__class__.__slots__ if hasattr(o, x))
        return s

    return sizeof(o) / 1024

import linecache

def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("#%s: %s:%s: %.1f KiB"
              % (index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def main():
    tracemalloc.start()
    ID = "4"

    events_filename = "config/Events/e-_800keV_0inc.root"
    Events = nesse.eventsFromG4root(events_filename)[:1]

    EF_filename = "config/Fields/4e10/NessieEF_Base4e7Linear0-150.0V.hf"
    WP_filename = "config/Fields/NessieWP_4e7Linear0-150V_grid.hf"

    Efield=nesse.fieldFromH5(EF_filename, rotate90=True)  

    weightingPotential = nesse.potentialFromH5(WP_filename, rotate90=True)

    print("Presim:")
    snapshot = tracemalloc.take_snapshot()
    display_top(snapshot)

    sim = nesse.Simulation("Example_sim", 125, Efield, _weightingPotential=weightingPotential, contacts=1)

    spiceFile= "g:/nesse/config/Spice/spice_step_New_1ns.csv"
    sim.setElectronicResponse(spiceFile)

    sim.setIDP(lambda x, y, z : float(ID) * 1e16)

    ef_bounds = [[axis[0],axis[-1]] for axis in Efield.grid]
    bounds = np.stack((ef_bounds[0],ef_bounds[1],[0,0.002]))
    sim.setBounds(bounds)

    print("Sim:")
    snapshot = tracemalloc.take_snapshot()
    display_top(snapshot)

    sim.setThreadNumber(6)

    sim.simulate(Events, ds=1e-6, diffusion=False, dt=1e-9, maxPairs=1, silence=True, parallel=False)

    print("Postsim:")
    snapshot = tracemalloc.take_snapshot()
    display_top(snapshot)
    sim.setWeightingField()

    print("Weighting Field:")
    snapshot = tracemalloc.take_snapshot()
    display_top(snapshot)

    sim.calculateInducedCurrent(Events, 1e-9, detailed=False)
 
    print(f"Events size: {total_size(Events)}")
    print(f"Event size: {total_size(Events[0])}")

    print("-----------------------------------")

    print("PostCurrent:")
    snapshot = tracemalloc.take_snapshot()
    display_top(snapshot)

    # sim.calculateElectronicResponse(Events)

    return None

if __name__ == "__main__":
    tracemalloc.start()
    t0 = time.time()
    main()
    t1 = time.time()
    print(tracemalloc.get_traced_memory())
    tracemalloc.stop()

    print(f"runtime: {t1-t0}")