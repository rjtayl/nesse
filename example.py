# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.getcwd()+"/src/")
import nesse
import numpy as np
import matplotlib.pyplot as plt
import csv
import cProfile
import pstats
from pstats import SortKey

from src.nesse.event import *

#example/testing of basic functionality

def main():
    #import G4 events
    events_filename = "config/Events/e-_800keV_0inc.root"
    Events = nesse.eventsFromG4root(events_filename)
    print("%d events loaded" %(len(Events)))

    #import SSD fields
    EF_filename = "config/Fields/NessieEF_4e7Linear0-150V_grid.hf"
    WP_filename = "config/Fields/NessieWP_4e7Linear0-150V_grid.hf"

    Efield=nesse.eFieldFromH5(EF_filename)  
    weightingPotential = nesse.weightingPotentialFromH5(WP_filename)

    print(weightingPotential,Efield)

    ef_bounds = [[axis[0],axis[-1]] for axis in Efield.grid]
    bounds = np.stack((ef_bounds[0],ef_bounds[1],[0,0.002]))
    
    #plot fields
    nesse.plot_field_lines(Efield,x_plane=True, density=2, show_plot=True)
    nesse.plot_potential(weightingPotential, bounds=bounds, x_plane=True, show_plot=True, mesh_size=(330,330))
    
    
    #create simulation
    sim = nesse.Simulation("Example_sim", Efield, weightingPotential)
    
    #import electronic response from spice
    spiceFile= "config/Spice/spice_step_New_1ns.csv"
    sim.setElectronicResponse(spiceFile)
    
    #set temperature    
    sim.setTemp(125)
    
    #set e-h drift boundaries 
    ef_bounds = [[axis[0],axis[-1]] for axis in Efield.grid]
    bounds = np.stack((ef_bounds[0],ef_bounds[1],[0,0.002]))
    sim.setBounds(bounds)
    
    #simulate events
    
    #simulate without diffusion
    i=5
    events = sim.simulate(Events[:i], eps=5e-5, dt=10e-9, interp3d=True, diffusion=True, plasma=False, maxPairs=10)
    sim.setWeightingField()
    sim.calculateInducedCurrent(events, 1e-10)
    sim.calculateElectronicResponse(events)
    
    #add noise
    for event in Events[:i]:
        event.addGaussianNoise(SNR=20)
    
    #plot induced current
    for event in Events[:i]:
        plt.plot(event.dt,event.dI, alpha=0.2)
    plt.show()
    
    #plot spice signals
    for event in Events[:i]:
        plt.plot(event.signal_times,event.signal_I)
    plt.show()

    #for event in Events[:i]:
    #    nessie.plot_event_drift(event, [[-0.001,0.001],[-0.001,0.001],[0,0.002]])
        
    

    #downsampling
    #nabPy_events = saveEventsNabPy(Events, "nabPyevents")
    
    #events_loaded = loadEventsNabPy("nabPyevents.pkl")

    

    print("Done")
    
if __name__ == "__main__":
    main()
