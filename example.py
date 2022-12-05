# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.getcwd()+"/src/")
import nessie
import numpy as np
import matplotlib.pyplot as plt

#example/testing of basic functionality

def main():
    #import G4 events
    events_filename = "config/Events/e-_800keV_0inc.root"
    Events = nessie.eventsFromG4root(events_filename)
    print("%d events loaded" %(len(Events)))

    #import SSD fields
    EF_filename = "config/Fields/NessieEF_Base4e7Linear0-150V.h5"
    WP_filename = "config/Fields/NessieWP_Base4e7Linear0-150V.h5"

    Efield=nessie.eFieldFromH5(EF_filename)
    weightingPotential = nessie.weightingPotentialFromH5(WP_filename)

    print(weightingPotential,Efield)
    
    #plot fields
    #nessie.plot_field_lines(Efield, Efield.bounds,x_plane=True, density=2, show_plot=False)
    
    #Note that the bounds should be the same but not necessarily the grid size. 
    
    
    #create simulation
    sim = nessie.Simulation("Example_sim", Efield, weightingPotential)
    sim.setTemp(125)
    bounds = np.stack((Efield.bounds[0],Efield.bounds[1],[0,0.002]))
    sim.setBounds(bounds)
    sim.setWeightingField()
    
    #simulate events (work in progress)
    
    #simulate without diffusion
    i=0
    sim.simulate(Events[:10], stepLimit=1000)
    #nessie.plot_event_drift(Events[i],[[-0.001,0.001],[-0.001,0.001],[0,0.002]])
    
    #simulate with diffusion
    #sim.simulate(Events[:10],eps=1e-5, stepLimit=1000, diffusion=True)
    #nessie.plot_event_drift(Events[i],[[-0.001,0.001],[-0.001,0.001],[0,0.002]],suffix="_diffusion")
    
    for event in Events:
        plt.plot(event.dt,event.dI, alpha=0.2)
    plt.show()
    
    #electronics

    #downsampling

    #add noise

    print("Done")
    
if __name__ == "__main__":
    main()
