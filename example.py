# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.getcwd()+"/src/")
import nessie
import numpy as np
import matplotlib.pyplot as plt
import csv
import cProfile
import pstats
from pstats import SortKey

from src.nessie.event import *

#example/testing of basic functionality

def main():
    #import G4 events
    events_filename = "config/Events/e-_800keV_0inc.root"
    Events = nessie.eventsFromG4root(events_filename)
    print("%d events loaded" %(len(Events)))

    #event = nessie.Event(0, np.array([[0, 0, 0], [0.001, 0, 0.0001]]), np.array([2000, 1000]), np.array([0, 0]))
    #event = nessie.Event(0, np.array([[0.001, 0, 0.001],]), np.array([40,]), np.array([0, ]))
    
    #n = -1
    #event0 = Events[0]
    #event = nessie.Event(0, event0.pos[:n], event0.dE[:n], event0.times[:n])
    #event = nessie.Event(0, np.array([event0.pos[n]]), np.array([event0.dE[n]]), np.array([event0.times[n]]))
    #event = Events[0]
    #Events = [event,]

    #import SSD fields
    EF_filename = "config/Fields/NessieEF_4e7Linear0-150V_grid.hf"
    WP_filename = "config/Fields/NessieWP_4e7Linear0-150V_grid.hf"

    Efield=nessie.eFieldFromH5(EF_filename)  
    weightingPotential = nessie.weightingPotentialFromH5(WP_filename)

    print(weightingPotential,Efield)
    
    #plot fields
    #nessie.plot_field_lines(Efield,x_plane=True, density=2, show_plot=True)
    #nessie.plot_potential(weightingPotential, Efield.bounds,x_plane=True, show_plot=True, mesh_size=(330,330))
    
    
    
    
    
    #create simulation
    sim = nessie.Simulation("Example_sim", Efield, weightingPotential)
    
    #import electronic response from spice
    spiceFile= "config/Spice/spice_step_New_1ns.csv"
    sim.setElectronicResponse(spiceFile)
    
    #set temperature    
    sim.setTemp(125)
    
    #set e-h drift boundaries 
    ef_bounds = [[axis[0],axis[-1]] for axis in Efield.grid]
    bounds = np.stack((ef_bounds[0],ef_bounds[1],[0,0.002]))
    sim.setBounds(bounds)
    
    #calculate weighting field
    #sim.setWeightingField()
    
    #nessie.plot_field_lines(sim.weightingField, Efield.bounds,x_plane=True, density=2, show_plot=True, log=False)
    
    #simulate events
    
    #simulate without diffusion
    i=5
    events = sim.simulate(Events[:i], eps=5e-5, dt=10e-9, interp3d=True, diffusion=True, plasma=False, maxPairs=10)
    #print(Events[0].quasiparticles)
    #print(Events[0].quasiparticles[0].pos)
    #print(Events[0].quasiparticles[0].time)
    #print(len(Events[0].quasiparticles[0].pos), len(Events[0].quasiparticles[0].time))
    sim.setWeightingField()
    sim.calculateInducedCurrent(events, 1e-10)
    sim.calculateElectronicResponse(events)
    #nessie.plot_event_drift(Events[0],[[-0.001,0.001],[-0.001,0.001],[0,0.002]])
    
    #cProfile.runctx('sim.simulate(Events[:i], eps=1e-4, interp3d=True, diffusion=True)', {'sim':sim, 'Events':Events, 'i':i},{}, 'sim_stats')
    
    #p = pstats.Stats('sim_stats')
    #p.sort_stats(SortKey.FILENAME).print_stats('charge_propagation.py')
    
    

    #print(len(event.pos_drift_e), len(event.pos_drift_e[0]), len(event.pos_drift_e[0][0]))
    
    #simulate with diffusion
    #sim.simulate(Events[:i],eps=1e-5, interp3d=True, diffusion=True)
    #nessie.plot_event_drift(Events[0],[[-0.001,0.001],[-0.001,0.001],[0,0.002]],suffix="_diffusion")
    
    #plot induced current
    for event in Events[:i]:
        plt.plot(event.dt,event.dI, alpha=0.2)
    plt.show()
    
    #plot spice signals
    for event in Events[:i]:
        plt.plot(event.signal_times,event.signal_I)
    plt.show()

    #downsampling
    #nabPy_events = saveEventsNabPy(Events, "nabPyevents")
    
    #events_loaded = loadEventsNabPy("nabPyevents.pkl")

    #add noise

    print("Done")
    
if __name__ == "__main__":
    main()
