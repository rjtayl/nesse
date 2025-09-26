# -*- coding: utf-8 -*-
import os
import sys
# sys.path.append(os.getcwd()+"/src/")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import nesse
import numpy as np

def test_CCE():
    #test "old" monoenergetic electron files

    ID = "4"

    Events = nesse.loadEvents("./tests/1kEvents_eventNum_0")

    EF_filename = "config/Fields/4e10/NessieEF_Base4e7Linear0-150.0V.hf"
    WP_filename = "config/Fields/NessieWP_4e7Linear0-150V_grid.hf"

    Efield=nesse.fieldFromH5(EF_filename, rotate90=True)  
    weightingPotential = nesse.potentialFromH5(WP_filename, rotate90=True)

    #first test that defauly hard CCE works
    sim = nesse.Simulation("Example_sim", 125, Efield, _weightingPotential=weightingPotential, contacts=1)



    assert list(map(lambda x: x.ID, events[:10])) == list(range(10)), f"Monoenergetic electron file event ID imported incorrectly"

    assert np.array_equal(saved_events[0].pos, events[0].pos), f"Monoenergetic electron file positions imported incorrectly."


if __name__ == "__main__":
    test_CCE() 