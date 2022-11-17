# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.getcwd()+"/src/")
import nessie

#example/testing of basic functionality

def main():
    #import G4 events
    events_filename = "config/Events/e-_800keV_0inc.root"
    Events = nessie.eventsFromG4root(events_filename)
    print("%d events loaded" %(len(Events)))

    #import SSD fields
    EF_filename = "config/Fields/NessieEF_Base4e7Linear0-150V.h5"
    WP_filename = "config/Fields/NessieWP_Base4e7Linear0-150V.h5"

    Ex,Ey,Ez=nessie.efFromH5(EF_filename)
    wp = nessie.wpFromH5(WP_filename)

    print(wp,Ex)

    #create simulation
    sim = nessie.Simulation("Example_sim", np.array([Ex,Ey,Ez]), wp)

    #simulate events

    #electronics

    #downsampling

    #add noise

if __name__ == "__main__":
    main()
