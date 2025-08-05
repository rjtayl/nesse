# -*- coding: utf-8 -*-
import os
import sys
# sys.path.append(os.getcwd()+"/src/")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import nesse
import numpy as np

def test_event():
    #test "old" monoenergetic electron files
    print(os.getcwd())
    filename = "./config/Events/e-_800keV_0inc.root"
    events = nesse.eventsFromG4root(filename)
    saved_events = nesse.loadEvents("./tests/e-_800keV_events")

    assert list(map(lambda x: x.ID, events[:10])) == list(range(10)), f"Monoenergetic electron file event ID imported incorrectly"

    assert np.array_equal(saved_events[0].pos, events[0].pos), f"Monoenergetic electron file positions imported incorrectly."

    assert np.array_equal(saved_events[0].times, events[0].times), f"Monoenergetic electron file times imported incorrectly."

    assert np.array_equal(saved_events[0].dE, events[0].dE), f"Monoenergetic electron file energies imported incorrectly."

    #test "new" monoenergetic proton files
    filename = "./config/Events/109Cd_Setup_e-_62.5keV.root"
    events = nesse.eventsFromG4root(filename, N=10)
    saved_events = nesse.loadEvents("./tests/62.5_109Cd_events")

    assert list(map(lambda x: x.ID, events[:2])) == list(map(lambda x: x.ID, saved_events)), f"Cd electron file event ID imported incorrectly, expected: {list(map(lambda x: x.ID, saved_events[:10]))}, found: {list(map(lambda x: x.ID, events[:10]))}"

    assert np.array_equal(saved_events[0].pos, events[0].pos), f"Cd electron file positions imported incorrectly."

    assert np.array_equal(saved_events[0].times, events[0].times), f"Cd electron file times imported incorrectly."

    assert np.array_equal(saved_events[0].dE, events[0].dE), f"Cd electron file energies imported incorrectly."



if __name__ == "__main__":
    test_event()  