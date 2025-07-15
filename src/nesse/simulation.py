import numpy as np
from itertools import compress
from .field import *
from .quasiparticles import *
from .charge_propagation import *
from tqdm.auto import tqdm
import csv
from .silicon import *
import copy
from .mobility import *
import os


import multiprocessing as mp

#TODO: right now all of the multiprocessing for simulation.py relies on first copying the objects, processing them,
# then returning a list of new objects. This isn't ideal for memory use, but I think the Event memory usage is 
#  sufficiently under control that we will save this to be improved in the future. 

def propagateCharge_helper(args):
    c, ds, dt, Ex_i, Ey_i, Ez_i, Emag_i, simBounds, temp, diffusion, NI, mobility_e, mobility_h = args
    return propagateCharge(c, ds, dt, Ex_i, Ey_i, Ez_i, Emag_i, simBounds,
                            temp, diffusion=diffusion, NI=NI,
                            mobility_e=mobility_e, mobility_h=mobility_h)

def default_impurity_concentration(x,y,z, IDP=1):
    return 1e16 * IDP

def calculateInducedCurrent_helper(args):
    event, dt, wf_interp, contact, detailed = args
    event.calculateInducedCurrent(dt, wf_interp, contact, detailed=detailed)
    return event

def calculateElectronicResponse_helper(args):
    event, electronicResponse, contacts = args
    for contact in contacts:
        event.convolveElectronicResponse(electronicResponse, contact)
    return event

class Simulation:
    '''
    Object contains all nesse objects needed to simulate a signal. 
    '''
    def __init__(self, _name, _temp, _electricField=None, _weightingPotential=None, _electricPotential=None,
                 _weightingField=None, _cceField=None, _chargeCaptureField=None, _electronicResponse=None, contacts=1,
                 _impurityConcentration= default_impurity_concentration, _mobility = [generalized_mobility_el, generalized_mobility_h]):        
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
        self.threads = os.cpu_count()

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
    
    def setThreadNumber(self, N):
        self.threads=N
        return None

    def setChargeCollectionEfficiency(self, type, depth=None, bounds=None, p0=None,p1=None, oxide_t=None):
        '''
        The primary purpose of this is to make a dead layer on the front face of the detector. We assume that the charge
        collection efficiency has been determined elsewhere (e.g. with GEANT), therefore NESSE does not account for 
        underdepleted detectors having a dead layer on the back of the detector. 

        We only use analytical models for a "hard" and "soft" dead layer. 
        '''
        #TODO: impliment soft dead layer
        if bounds is None:
            bounds = self.bounds

        if type == "hard":
            #The edge of the detector should effectively be a dead layer always
            d = bounds[2][0] if depth is None else depth

            self.cceField = lambda x,y,z: 0 if z<=d or z>=bounds[2][1] else 1
        
        if type == "soft":
            #by default ignore the oxide layer contribution
            if oxide_t is None:
                p1=0 if p1 is None else p1
                self.cceField = lambda x,y,z: 1-(p1-1)*np.exp(-z/depth) if z > bounds[2][0] else 0
            else:
                p0=0 if p0 is None else p0
                p1=0 if p1 is None else p1
                self.cceField = lambda x,y,z: 1-(p1-1)*np.exp(-(z-oxide_t)/depth) if z > oxide_t else p0
        
        #TODO: importing user models

        return None

    def setChargeCaptureField(self):
        return None

    def setElectronicResponse(self, spiceFile=None, t1=None,t2=None, length=7000):
        '''
        This either takes spice output csv file or assumes a CR-RC signal shaping and the user has to specify each time constant
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
            step = t1/(t1-t2) * (np.exp(-ts/t1)-np.exp(-ts/t2))
            self.electronicResponse = {"times":ts, "step":step}

        return None

    def simulate(self, events, ds, dt, coulomb=False, diffusion=False, capture=False, d=None, interp3d = True, maxPairs=100, 
                Efield=None, bounds=None, silence=False, parallel=False, detailed=False):
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

        if self.cceField is None:
            self.setChargeCollectionEfficiency("hard")

        #Find electron and hole drift paths for each event
        for i in (t:=tqdm(range(len(events)))):
            t.set_description(f"Drift Calculation", refresh=True)
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

                # check if charge is collected using CCE Field, if not we don't generate it. This is for the simple dead
                # layer models, if using carrier lifetime models, rely on tauTrap instead.
                CCE = self.cceField(*(event.pos[j]))
                if CCE == 0: continue
                else:
                    pairNr = int(pairs[j])
                    factor = 1
                    if pairNr > maxPairs:
                        factor = pairNr/maxPairs
                        pairNr = maxPairs

                    pairNr = int(np.round(CCE*pairNr))

                    # TODO: Provide a radius to smooth initial charge cloud (currently 0)
                    cc_e = cc_e + initializeChargeCloud(-factor*qe_SI, factor*me_SI, pairNr, event.times[j], 0, event.pos[j])
                    cc_h = cc_h + initializeChargeCloud(factor*qe_SI, factor*me_SI, pairNr, event.times[j], 0, event.pos[j])

            cc = cc_e + cc_h

            alive = np.ones(len(cc)) == 1

            t.set_postfix_str(f"Event {i}, Total quasiparticles: {len(cc)}", refresh=True)
            
            if coulomb == False and parallel == True:
                '''This is still fairly experimental. Note that you cannot have ANY lambda functions as an argument
                   or else the pickling for the Pool will not work. This is why impurity concentration now has a global
                   function outside the class.'''
                args_list = [
                    (c, ds, dt, Ex_i, Ey_i, Ez_i, Emag_i, simBounds,
                     self.temp, diffusion, self.impurityConcentration,
                     self.mobility[0], self.mobility[1])
                    for c in cc
                ]

                with mp.Pool(processes=int(self.threads)) as pool:
                    cc = pool.map(propagateCharge_helper, args_list)
                    
            else:
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

            #before returning particles, compress the data by changing from python list to numpy array
            for charge in cc:
                charge.compressData()
    
            event.quasiparticles = cc

        return events

    # TODO: If contacts are same geometry use same interpolation but shift positions? 
    def calculateInducedCurrent(self, events, dt, contacts = None, interp3d=True, parallel=False, detailed=False):
        if contacts is None: contacts=np.arange(self.contacts)

        if parallel == False:
            for i_contact in range(len(contacts)):
                contact= contacts[i_contact]
                
                weightingField = self.weightingPotential[contact].toField()
                weightingFieldx_interp, weightingFieldy_interp, weightingFieldz_interp, weightingFieldMag_interp = weightingField.interpolate(interp3d)
                wf_interp = [weightingFieldx_interp, weightingFieldy_interp, weightingFieldz_interp, weightingFieldMag_interp]
                
                del weightingField
            
                for event in events:
                    if i_contact == len(contacts)-1:
                        event.calculateInducedCurrent(dt, wf_interp, contact, detailed=detailed)
                    else:
                        event.calculateInducedCurrent(dt, wf_interp, contact, detailed=True)


        elif parallel==True:
            
            for i_contact in range(len(contacts)):
                contact= contacts[i_contact]

                weightingField = self.weightingPotential[contact].toField()
                weightingFieldx_interp, weightingFieldy_interp, weightingFieldz_interp, weightingFieldMag_interp = weightingField.interpolate(interp3d)
                wf_interp = [weightingFieldx_interp, weightingFieldy_interp, weightingFieldz_interp, weightingFieldMag_interp]
                
                del weightingField

                if i_contact == len(contacts)-1:
                    args_list = [(event, dt, wf_interp, contact, detailed) for event in events]
                else:
                    args_list = [(event, dt, wf_interp, contact, True) for event in events]
                
                with mp.Pool(processes=self.threads) as pool:
                    results = pool.map(calculateInducedCurrent_helper, args_list)

                # Update the original events list with the processed events
                for i, event in enumerate(results):
                    events[i] = event

    def calculateElectronicResponse(self, events, contacts = None, parallel=False):
        if contacts is None: contacts=np.arange(self.contacts)

        if parallel:
            args_list = [(event, self.electronicResponse, contacts) for event in events]
            with mp.Pool(processes=self.threads) as pool:
                results = pool.map(calculateElectronicResponse_helper, args_list)

            for i, event in enumerate(results):
                    events[i] = event

        else:
            for event in events:
                for contact in contacts:
                    event.convolveElectronicResponse(self.electronicResponse, contact)

    def getSignals(self, events, dt, contacts = None, interp3d=True, parallel=False):
        self.calculateInducedCurrent(events, dt, contacts, interp3d, parallel=parallel)
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