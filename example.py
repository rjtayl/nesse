# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.getcwd()+"/src/")
import nessie

def main():
    #import G4 events

    #import SSD fields
    EF_filename = "config/Fields/NessieEF_Base4e7Linear0-150V.h5"
    WP_filename = "config/Fields/NessieWP_Base4e7Linear0-150V.h5"

    Ex,Ey,Ez=nessie.efFromH5(EF_filename)
    wp = nessie.wpFromH5(WP_filename)

    print(wp,Ex)

    #create simulation

    #simulate events

    #electronics

    #downsampling

    #add noise

if __name__ == "__main__":
    main()
