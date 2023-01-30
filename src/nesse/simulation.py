import numpy as np
from itertools import compress
from .field import *
from .quasiparticles import *
from scipy.interpolate import RegularGridInterpolator 
from .charge_propagation import *
from tqdm import tqdm
import csv
from .silicon import *

#from line_profiler import LineProfiler

#from joblib import Parallel, delayed
#import multiprocessing

#num_cores = multiprocessing.cpu_count()

class Simulation:
    '''
    Object contains all nessie objects needed to simulate a signal. 
    Currently assumes only one contact.
    
    '''
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

    def simulate(self, events, eps, dt, plasma=False, diffusion=False, capture=False, d=None, interp3d = True, maxPairs=100):
        '''
        Where it all happens! 
        When calling this function you determine which effects you want to simulate (eg. plasma, diffusion, etc.)
        Reduce maxPairs to run faster by having fewer quasiparticles with greater charge.
        Setting interp3d to true here will use our cython interpolator throughout the simulation.
        For machines with many cores you can try setting parallel=True. We see slight speedups but your mileage may vary. 
        '''
    
        # Get electric field interpolations
        Ex_i, Ey_i, Ez_i, Emag_i = self.electricField.interpolate(interp3d)
        
        simBounds = self.bounds if self.bounds is not None else [[axis[0],axis[-1]] for axis in self.electricField.grid]

        #Find electron and hole drift paths for each event
        for i in tqdm(range(len(events))):
            event = events[i]
            #rint will give an integer number of e-h pairs, but will need to use Fano factor for a proper calculation
            pairs = np.rint(event.dE/ephBestFit(self.temp))

            # electron charge cloud assembly
            cc_e = []
            # hole charge cloud assembly
            cc_h = []

            # Add charge cloud parts at each position where energy was deposited
            for j in tqdm(range(len(event.pos))):
                pairNr = int(pairs[j])
                factor = 1
                if pairNr > maxPairs:
                    factor = pairNr/maxPairs
                    pairNr = maxPairs

                # TODO: Provide a radius to smooth initial charge cloud (currently 0)
                cc_e = cc_e + initializeChargeCloud(-factor*qe_SI, factor*me_SI, pairNr, event.times[j], 0, event.pos[j])
                cc_h = cc_h + initializeChargeCloud(factor*qe_SI, factor*me_SI, pairNr, event.times[j], 0, event.pos[j])

            cc = cc_e + cc_h

            alive = np.ones(len(cc)) == 1
            print("Total quasiparticles: %d" % len(cc))

            #lp = LineProfiler()
            #lp.add_function(generalized_mobility_el)
            #lp_wrapper = lp(updateQuasiParticles)

            # Loop over alive particles until all have been stopped/collected
            pbar = tqdm()
            while np.any(alive):
                cc_new = updateQuasiParticles(list(compress(cc, alive)), eps, dt, Ex_i, Ey_i, Ez_i, Emag_i, simBounds, self.temp, diffusion=diffusion, coulomb=plasma)
                #cc_new = lp_wrapper(list(compress(cc, alive)), eps, dt, Ex_i, Ey_i, Ez_i, Emag_i, simBounds, self.temp, diffusion=diffusion, coulomb=plasma)
                counter = 0
                for j in range(len(alive)):
                    if alive[j]:
                        cc[j] = cc_new[counter]
                        counter+=1

                alive = np.array([o.alive for o in cc])
                pbar.update(1)

            event.quasiparticles = cc
            #lp.print_stats()
        return events

    def calculateInducedCurrent(self, events, dt):
        for event in events:
            event.calculateInducedCurrent(self.weightingField, dt, interp3d=True)

    def calculateElectronicResponse(self, events):
        for event in events:
            event.convolveElectronicResponse(self.electronicResponse)


        '''#get induced current
        print("calculating induced current")
        
        if self.weightingField is None: self.setWeightingField()
        
        for event in events:
            event.calculateInducedCurrent(self.weightingField, 0.1e-9, interp3d=interp3d)
            
        if self.electronicResponse is not None:
                for event in events:
                    event.convolveElectronicResponse(self.electronicResponse)'''

