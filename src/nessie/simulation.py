import numpy as np
from .field import *
from scipy.interpolate import RegularGridInterpolator 
from .charge_propagation import *
from tqdm import tqdm
import csv
from .silicon import *

class Simulation:
    def __init__(self, _name, _electricField=None, _weightingPotential=None, _electricPotential=None, _weightingField=None,  
                    _cceField=None, _chargeCaptureField=None, _electronicResponse=None, _temp=None):        
        self.name = _name
        self.electricField = _electricField
        self.electricPotential = _electricPotential
        self.weightingPotential = _weightingPotential
        self.weightingField = _weightingField
        self.cceField = _cceField
        self.chargeCaptureField = _chargeCaptureField
        self.electronicResponse = _electronicResponse
        self.temp = _temp
        self.bounds = None
        
    def __str__(self):
       return ("Nessie Simulation object \n Name: %s \n Electric Field: %s \n Weighting Potential: %s \n CCE Field: %s \n Charge Capture Field: %s \n Electronic Response: %s \n"
                % (self.name, self.electricField,self.weightingPotential, self.cceField, self.chargeCaptureField,          
                   self.electronicResponse))
    
    def setTemp(self,T):
        self.temp = T
        return None
        
    def setBounds(self, bounds):
        self.bounds = bounds
        return None
    
    def setElectricField(self):
        grid = self.electricPotential.grid
        gradient = np.gradient(self.electricPotential.data, grid[0],grid[1],grid[2])
        try:
            self.electricField = Field("Electric Field", gradient[0],gradient[1],gradient[2], self.electricPotential.gri)
        except:
            print("No electric potential from which to calculate weighting field.")
        return None

    def setWeightingField(self):
        grid = self.weightingPotential.grid
        gradient = np.gradient(self.weightingPotential.data, grid[0],grid[1],grid[2])
        try:
            self.weightingField = Field("Weighting Field", gradient[0],gradient[1],gradient[2], self.weightingPotential.grid)
        except:
            print("No weighting potential from which to calculate weighting field.")
        return None

    def setChargeCollectionEfficiencyField(self):
        return None

    def setChargeCaptureField(self):
        return None

    def setElectronicResponse(self, spiceFile):
        ts = []
        step = []
        with open(spiceFile,'r') as csvfile:
            plots = csv.reader(csvfile, delimiter=',')
            for row in plots:
                ts.append(float(row[0])*1e-9) #convert from ns to s
                step.append(float(row[1]))
        self.electronicResponse = {"times":ts, "step":step}
        return None

    def simulate(self, events, eps=1e-4, plasma=False, diffusion=False, capture=False, stepLimit=1000, d=None):
    
        # Get electric field interpolations
        eFieldx_interp, eFieldy_interp, eFieldz_interp, eFieldMag_interp = self.electricField.interpolate()
        #eFieldx_interp = lambda x: [0,]
        #eFieldy_interp = lambda x: [0,]
        #eFieldz_interp = lambda x: [750e2,]
        #eFieldMag_interp = lambda x: [750e2,]
        
        simBounds = self.bounds if self.bounds is not None else [[axis[0],axis[-1]] for axis in self.electricField.grid]
        
        #Find electron and hole drift paths for each event
        print("drifting events:")
        for i in range(len(events)):
            print("Event %i" % (i))
            event=events[i]
            new_pos = []
            new_times = []
            new_pos_h = []
            new_times_h = []
            #rint will give an integer number of e-h pairs, but will need to use Fano factor for a proper calculation
            pairs = np.rint(event.dE/ephBestFit(self.temp))

            for j in tqdm(range(len(event.pos))):
                print(pairs[j])
                for k in tqdm(range(int(pairs[j]))):
                    x,y,z,t = propagateCarrier(event.pos[j][0], event.pos[j][1], event.pos[j][2], eps, eFieldx_interp, eFieldy_interp, 
                                            eFieldz_interp, eFieldMag_interp, simBounds, self.temp,d=d, stepLimit=stepLimit, diffusion=diffusion)
                    new_pos.append(np.stack((x,y,z), axis=-1))
                    new_times.append(t)
                
                    x_h,y_h,z_h,t_h = propagateCarrier(event.pos[j][0], event.pos[j][1], event.pos[j][2], eps, eFieldx_interp, eFieldy_interp, 
                                            eFieldz_interp, eFieldMag_interp, simBounds, self.temp,d=d, stepLimit=stepLimit, diffusion=diffusion,electron=False)
                    new_pos_h.append(np.stack((x_h,y_h,z_h), axis=-1))
                    new_times_h.append(t_h)
                
            event.setDriftPaths(new_pos,new_times)
            event.setDriftPaths(new_pos_h,new_times_h,electron=False)
            
            #get drift velocities
            event.getDriftVelocities()
        
        #get induced current
        print("calculating induced current")
        
        if self.weightingField is None: setWeightingField()
        
        for event in events:
            event.calculateInducedCurrent(self.weightingField, 0.1e-9)
            
        if self.electronicResponse is not None:
                for event in events:
                    event.convolveElectronicResponse(self.electronicResponse)
            
        return None
