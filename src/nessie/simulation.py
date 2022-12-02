import numpy as np
from .field import *
from scipy.interpolate import RegularGridInterpolator 
from .charge_propagation import *

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
        try:
            self.electricField = Field("Electric Field", *np.gradient(self.electricPotential.data), self.electricPotential.bounds)
        except:
            print("No electric potential from which to calculate weighting field.")
        return None

    def setWeightingField(self):
        try:
            self.weightingField = Field("Weighting Field", *np.gradient(self.weightingPotential.data), self.weightingPotential.bounds)
        except:
            print("No weighting potential from which to calculate weighting field.")
        return None

    def setChargeCollectionEfficiencyField(self):
        return None

    def setChargeCaptureField(self):
        return None

    def setElectronicResponse(self):
        return None

    def simulate(self, events, eps=1e-4, plasma=False, diffusion=False, capture=False, stepLimit=1000, d=None):
        eFieldBounds = self.electricField.bounds
        eFieldShape = np.shape(self.electricField.fieldx)
        x = np.linspace(eFieldBounds[0][0],eFieldBounds[0][1],eFieldShape[0])
        y = np.linspace(eFieldBounds[1][0],eFieldBounds[1][1],eFieldShape[1])
        z = np.linspace(eFieldBounds[2][0],eFieldBounds[2][1],eFieldShape[2])
        
        eFieldx_interp  = RegularGridInterpolator((x,y,z),self.electricField.fieldx)
        eFieldy_interp  = RegularGridInterpolator((x,y,z),self.electricField.fieldy)
        eFieldz_interp  = RegularGridInterpolator((x,y,z),self.electricField.fieldz)
        
        eFieldMag = np.sqrt(self.electricField.fieldx**2+self.electricField.fieldy**2+self.electricField.fieldz**2)
        eFieldMag_interp = RegularGridInterpolator((x,y,z),eFieldMag)
        
        simBounds = self.bounds if self.bounds is not None else eFieldBounds
        
        for event in events:
            new_pos = []
            new_times = []
            new_pos_h = []
            new_times_h = []
            for i in range(len(event.pos)):
                print(i)
                x,y,z,t = propagateCarrier(event.pos[i][0], event.pos[i][1], event.pos[i][2], eps, eFieldx_interp, eFieldy_interp, 
                                        eFieldz_interp, eFieldMag_interp, simBounds, self.temp,d=d, stepLimit=stepLimit, diffusion=diffusion)
                new_pos.append(np.stack((x,y,z), axis=-1))
                new_times.append(t)
            
                x_h,y_h,z_h,t_h = propagateCarrier(event.pos[i][0], event.pos[i][1], event.pos[i][2], eps, eFieldx_interp, eFieldy_interp, 
                                        eFieldz_interp, eFieldMag_interp, simBounds, self.temp,d=d, stepLimit=stepLimit, diffusion=diffusion,electron=False)
                new_pos_h.append(np.stack((x_h,y_h,z_h), axis=-1))
                new_times_h.append(t_h)
                
            event.setDriftPaths(new_pos,new_times)
            event.setDriftPaths(new_pos_h,new_times_h,electron=False)
            
        return None
