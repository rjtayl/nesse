import numpy as np
from itertools import compress
from .field import *
from .quasiparticles import *
from scipy.interpolate import RegularGridInterpolator 
from .charge_propagation import *
from tqdm.auto import tqdm
import csv
from .silicon import *
import copy
from .mobility import *

#from line_profiler import LineProfiler

#from joblib import Parallel, delayed
#import multiprocessing

#num_cores = multiprocessing.cpu_count()

class Simulation:
    '''
    Object contains all nessie objects needed to simulate a signal. 
    Currently assumes only one contact.
    
    '''
    def __init__(self, _name, _temp, _electricField=None, _weightingPotential=None, _electricPotential=None,
                 _weightingField=None, _cceField=None, _chargeCaptureField=None, _electronicResponse=None, contacts=1,
                 _impurityConcentration= lambda x, y, z : 1e16, _mobility = [generalized_mobility_el, generalized_mobility_h]):        
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
        self.contacts = contacts
        self.impurityConcentration = _impurityConcentration
        self.mobility = _mobility

        if contacts is not None and type(_weightingPotential) is not list:
            centers = find_centers(contacts)
            wps = []
            for i in range(contacts):
                wp_temp = copy.deepcopy(_weightingPotential)
                wp_temp.shift(centers[i]+(0,))
                wps.append(wp_temp)
            self.weightingPotential=wps

        if _electricField is None and _electricPotential is not None:
            self.electricField = _electricPotential.toField()
            
    def __str__(self):
       return ("Nessie Simulation object \n Name: %s \n Electric Field: %s \n Weighting Potential: %s \n CCE Field: %s \n Charge Capture Field: %s \n Electronic Response: %s \n"
                % (self.name, self.electricField,self.weightingPotential, self.cceField, self.chargeCaptureField,          
                   self.electronicResponse))
    
    def setTemp(self,T):
        self.temp = T
        return None
    
    def setMobility(self,mobility_e, mobility_h):
        self.mobility = [mobility_e, mobility_h]
        return None
    
    def setIDP(self, IDP): #IDP(x,y,z)->NI[m^-3]
        if type(IDP) is Potential:
            self.impurityConcentration = IDP.interpolate()

        else:
            self.impurityConcentration = IDP
        return None
        
    def setBounds(self, bounds):
        self.bounds = bounds
        return None
    
    def setElectricField(self):
        grid = self.electricPotential.grid
        gradient = np.gradient(self.electricPotential.data, grid[0],grid[1],grid[2])
        try:
            self.electricField = Field("Electric Field", gradient[0],gradient[1],gradient[2], self.electricPotential.grid)
        except:
            print("No electric potential from which to calculate weighting field.")
        return None
    
    def recenter_hex_contacts(self, R, s):
        centers = find_centers(self.contacts, R, s)
        for i in range(centers):
            self.weightingPotential[i].shift(centers[i]+(0,))
    
    def setWeightingField(self):
        contacts = range(self.contacts)
        try:
            self.weightingField = [self.weightingPotential[contact].toField() for contact in contacts]
        except:
            print("No weighting potential from which to calculate weighting field.")
        return None

    def setChargeCollectionEfficiencyField(self):
        return None

    def setChargeCaptureField(self):
        return None

    def setElectronicResponse(self, spiceFile=None, t1=None,t2=None, length=7000):
        '''
        This either takes spice output csv file or assumes a RC-CR signal shaping and the user has to specify each time constant
        approximate values for Nab are 5 us and 7 ns
        '''
        if spiceFile is not None:
            ts = []
            step = []
            with open(spiceFile,'r') as csvfile:
                plots = csv.reader(csvfile, delimiter=',')
                for row in plots:
                    ts.append(float(row[0])*1e-9) #convert from ns to s
                    step.append(float(row[1]))
            self.electronicResponse = {"times":ts, "step":step}
        
        elif t1 is not None and t2 is not None:
            ts = np.arange(length)*1e-9
            step = 1/(t1-t2) * (np.exp(-ts/t1)-np.exp(-ts/t2))
            step = (np.exp(-ts/t1)-np.exp(-ts/t2))
            self.electronicResponse = {"times":ts, "step":step}

        return None

    def simulate(self, events, ds, dt, coulomb=False, diffusion=False, capture=False, d=None, interp3d = True, maxPairs=100, 
                Efield=None, bounds=None, silence=False):
        '''
        Where it all happens! 
        When calling this function you determine which effects you want to simulate (eg. plasma, diffusion, etc.)
        Reduce maxPairs to run faster by having fewer quasiparticles with greater charge.
        Setting interp3d to true here will use our cython interpolator throughout the simulation.
        For machines with many cores you can try setting parallel=True. We see slight speedups but your mileage may vary. 
        '''
    
        # Get electric field interpolations
        if Efield is None:
            Ex_i, Ey_i, Ez_i, Emag_i = self.electricField.interpolate(interp3d)
        else:
            Ex_i, Ey_i, Ez_i, Emag_i= Efield
        
        if bounds is None:
            simBounds = self.bounds if self.bounds is not None else [[axis[0],axis[-1]] for axis in self.electricField.grid]
        else: 
            simBounds = bounds

        #Find electron and hole drift paths for each event
        for i in tqdm(range(len(events))):
            event = events[i]
            #rint will give an integer number of e-h pairs, then adjust for variance using sigma = sqrt(FN)
            # assumes in linear regeme of fano factor
            pairs_0 = event.dE/ephBestFit(self.temp)
            pairs = np.rint(np.random.normal(pairs_0,np.sqrt(Fano_Si*pairs_0))) 

            # electron charge cloud assembly
            cc_e = []
            # hole charge cloud assembly
            cc_h = []

            # Add charge cloud parts at each position where energy was deposited
            for j in tqdm(range(len(event.pos)), disable=silence):
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

            # Loop over alive particles until all have been stopped/collected
            pbar = tqdm(disable=silence)
            while np.any(alive):
                cc_new = updateQuasiParticles(list(compress(cc, alive)), ds, dt, Ex_i, Ey_i, Ez_i, Emag_i, simBounds,
                                              self.temp, diffusion=diffusion, coulomb=coulomb,NI=self.impurityConcentration,
                                              mobility_e = self.mobility[0], mobility_h = self.mobility[1])
                
                counter = 0
                for j in range(len(alive)):
                    if alive[j]:
                        cc[j] = cc_new[counter]
                        counter+=1

                alive = np.array([o.alive for o in cc])
                pbar.update(1)

            event.quasiparticles = cc

        return events

    def calculateInducedCurrent(self, events, dt, contacts = None, interp3d=True):
        if contacts is None: contacts=np.arange(self.contacts)
        if self.weightingField is None: self.setWeightingField()
        for contact in contacts:
            weightingFieldx_interp, weightingFieldy_interp, weightingFieldz_interp, weightingFieldMag_interp = self.weightingField[contact].interpolate(interp3d)
            wf_interp = [weightingFieldx_interp, weightingFieldy_interp, weightingFieldz_interp, weightingFieldMag_interp]
        
            for event in events:
                event.calculateInducedCurrent(dt, wf_interp, contact)

    def calculateInducedCharge(self, events, contacts = None):
        if contacts is None: contacts=np.arange(self.contacts)
        for contact in contacts:
            for event in events:
                event.calculateIntegratedCharge(contact)

    def calculateElectronicResponse(self, events, contacts = None):
        if contacts is None: contacts=np.arange(self.contacts)
        for event in events:
            for contact in contacts:
                event.convolveElectronicResponse(self.electronicResponse, contact)

    def getSignals(self, events, dt, contacts = None, interp3d=True):
        self.calculateInducedCurrent(events, dt, contacts, interp3d)
        self.calculateElectronicResponse(events, contacts)

def find_centers(N,R=5.15e-3,s=0.1e-3):    
    ''' This functions finds where to place the centers of hexagonal configuration depending on the hexagon radius 
        and seperation between hexagons. Default values are correct for Nab detectors. This is here primarily to copy
        over weighting potentials for multiple contact sims. '''
    
    centers = [(0,0)]
    o_centers = [(0,0)]
    
    r = (np.sqrt(3)*R+s)
    theta = np.radians(np.arange(0,360,60))
    dx = np.round(r*np.cos(theta), 11)
    dy = np.round(r*np.sin(theta),11)
    
    while len(centers) < N:
        new_centers = []
        for center in o_centers:
            oldx, oldy = center
            for i in range(len(theta)):
                new_center = (np.round(oldx+dx[i], 8), np.round(oldy+dy[i], 8))
                if new_center not in centers:
                    new_centers.append(new_center)
                    centers.append(new_center)
        o_centers = new_centers
    return centers[:N]