# -*- coding: utf-8 -*-
import os
import sys
# sys.path.append(os.getcwd()+"/src/")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import nesse
import numpy as np

def test_event():
    #test "old" monoenergetic electron files
    filename = "./config/Events/e-_800keV_0inc.root"
    events = nesse.eventsFromG4root(filename)
    saved_events = nesse.loadEvents("./tests/e-_800keV_events")

    assert list(map(lambda x: x.ID, events[:10])) == list(range(10)), f"Monoenergetic electron file event ID imported incorrectly"

    assert np.array_equal(saved_events[0].pos, events[0].pos), f"Monoenergetic electron file positions imported incorrectly."

    assert np.array_equal(saved_events[0].times, events[0].times), f"Monoenergetic electron file times imported incorrectly."

    assert np.array_equal(saved_events[0].dE, events[0].dE), f"Monoenergetic electron file energies imported incorrectly."

    #test "new" monoenergetic electron files
    filename = "./config/Events/109Cd_Setup_e-_62.5keV.root"
    events = nesse.eventsFromG4root(filename, N=10)
    saved_events = nesse.loadEvents("./tests/62.5_109Cd_events")

    assert list(map(lambda x: x.ID, events[:2])) == list(map(lambda x: x.ID, saved_events)), f"Cd electron file event ID imported incorrectly, expected: {list(map(lambda x: x.ID, saved_events[:10]))}, found: {list(map(lambda x: x.ID, events[:10]))}"

    assert np.array_equal(saved_events[0].pos, events[0].pos), f"Cd electron file positions imported incorrectly."

    assert np.array_equal(saved_events[0].times, events[0].times), f"Cd electron file times imported incorrectly."

    assert np.array_equal(saved_events[0].dE, events[0].dE), f"Cd electron file energies imported incorrectly."

    #test "NabSim" event files
    filename = "./config/Events/1kEvents_eventNum_0.root"
    events = nesse.eventsFromG4root(filename, N=10, nab_file=True)
    saved_events = nesse.loadEvents("./tests/1kEvents_eventNum_0")

    assert list(map(lambda x: x.ID, events)) == list(map(lambda x: x.ID, saved_events)), f"Nabsim file event IDs imported incorrectly, expected: {list(map(lambda x: x.ID, saved_events))}, found: {list(map(lambda x: x.ID, events))}"

    assert np.array_equal(saved_events[0].pos, events[0].pos), f"Nabsim electron positions imported incorrectly."

    assert np.array_equal(saved_events[0].times, events[0].times), f"Nabsim electron times imported incorrectly."

    assert np.array_equal(saved_events[0].dE, events[0].dE), f"Nabsim electron energies imported incorrectly."

    assert np.array_equal(saved_events[1].pos, events[1].pos), f"Nabsim proton positions imported incorrectly."

    assert np.array_equal(saved_events[1].times, events[1].times), f"Nabsim proton times imported incorrectly."

    assert np.array_equal(saved_events[1].dE, events[1].dE), f"Nabsim proton energies imported incorrectly."

    assert list(map(lambda x: x.detector, events)) == list(map(lambda x: x.detector, saved_events)), f"Nabsim detector assigned improperly."



if __name__ == "__main__":
    test_event()  