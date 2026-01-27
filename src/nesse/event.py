import numpy as np
from .constants import *
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
from tqdm import tqdm
import pickle
import gc

##########
# 

##########

class Event:
    '''
    The Event class contains all information about a single particle event in the detector. 
    This starts with information of energy depositions imported from Geant4 (pos,dE,times). 
    Electron and hole drift paths are stored and used to determine the signal on a detector. 

    Currently this only works for a single contact, but we plan to extend to all contacts.
    '''
    def __init__(self, _id, _pos, _dE, _times, _PID=None, _detector="Upper"):
        self.ID = _id # This refers to a geant/decay event.
        self.pos = _pos
        self.dE = _dE
        self.times = _times

        #TODO: use standard particle ID for monte carlo from Particle Data Group (e- = 11, p = 2212)
        # for now just take whatever geant says.
        self.PID = _PID

        # assumes "Upper" detector, meaning oriented in positive z direction. This tag is for keeping track of nabsim
        # events even after rotation and shifting for nesse simulation. 
        self.detector = _detector 

        self.dQ = {}
        self.dI = {}
        self.dt = {}

        self.quasiparticles = []
        
        self.signal = {}
        self.signal_times = {}

    def __repr__(self):
        return f"Event(ID:{self.ID}, PID:{self.PID}, Energy:{round(np.sum(self.dE))}, QPs:{len(self.quasiparticles)}, Det: {self.detector})"

    def clearQP(self):
        self.quasiparticles = []
        gc.collect()

    def shift_pos(self,new_pos=[0,0,0]):
        '''
        Shifts position to specefied new_pos. 
        TODO: users may expect to give the shift rather than the location, consider changing.
        '''
        shift = np.array(new_pos) - self.pos[0]
        self.pos = self.pos + shift
        return None
    
    def rotate_pos(self, rotation=[0,0,0]):
        ''' 
        Euler roation of all event positions in the x-convention (zxz) in degrees. 
        This is intended to be used to rotate lower detector events so that they line up with the same fields used for
        the top detector, but is kept general for other potential use cases. 
        '''
        #TODO: test speed maybe move over to cython
        from scipy.spatial.transform import Rotation as R
        r = R.from_euler('zxz', rotation, degrees=True)
        self.pos = r.apply(self.pos)

        return None
        
    def convolveElectronicResponse(self, electronicResponse, contact=0):
        func_I = interp1d(self.dt[contact], self.dI[contact], bounds_error=False, fill_value=0)
        temp_times = electronicResponse["times"]
        dt = np.diff(temp_times)[0]
        
        self.signal[contact] = np.convolve(func_I(temp_times),electronicResponse["step"])[:max(len(electronicResponse["times"]),
                                                                                               len(temp_times))]
        self.signal_times[contact] = np.arange(0,len(self.signal[contact])*dt, dt)
        
        return None
        
    def addGaussianNoise(self, sigma=1, SNR=None):
        if SNR is not None:
            sigma = np.max(np.abs(self.signal))/SNR
        self.signal += np.random.normal(0, sigma , len(self.signal))   
        
    def convertUnits(self, energyConversionFactor, lengthConversionFactor, timeConversionFactor):
        self.dE *= energyConversionFactor
        self.pos *= lengthConversionFactor
        self.times *= timeConversionFactor
        return None
        
    def calculateInducedCurrent(self, dt, WF_interp, contact=0, detailed=False):
        weightingFieldx_interp, weightingFieldy_interp, weightingFieldz_interp, weightingFieldMag_interp = WF_interp
        
        max_time = max([o.time[-1] for o in self.quasiparticles])
        start_time = min([o.time[0] for o in self.quasiparticles])
        times_I = np.arange(start_time,max_time,dt)
        
        induced_I = np.zeros(len(times_I))

        if detailed:
            for i in range(len(self.quasiparticles)):
                o = self.quasiparticles[i]
                
                Is = o.q * np.sum(o.vel * np.array((weightingFieldx_interp(o.pos),
                            weightingFieldy_interp(o.pos),
                            weightingFieldz_interp(o.pos))).T, axis=1)

                if len(Is) > 0:
                    func_I = interp1d(o.time, Is, bounds_error=False, fill_value=0)
                    induced_I += func_I(times_I)
        else:
            while len(self.quasiparticles) > 0:
                o = self.quasiparticles.pop()
                
                Is = o.q * np.sum(o.vel * np.array((weightingFieldx_interp(o.pos),
                            weightingFieldy_interp(o.pos),
                            weightingFieldz_interp(o.pos))).T, axis=1)

                if len(Is) > 0:
                    func_I = interp1d(o.time, Is, bounds_error=False, fill_value=0)
                    induced_I += func_I(times_I)

        
        self.dI[contact] = induced_I
        self.dt[contact]=times_I-start_time
        
        return None    
    
    def calculateInducedCurrent_eh(self, dt, WF_interp, electron = True, detailed=False):
        weightingFieldx_interp, weightingFieldy_interp, weightingFieldz_interp, weightingFieldMag_interp = WF_interp
        
        max_time = max([o.time[-1] for o in self.quasiparticles])
        start_time = min([o.time[0] for o in self.quasiparticles])
        times_I = np.arange(start_time,max_time,dt)
        
        induced_I = np.zeros(len(times_I))

        ehs = [o for o in self.quasiparticles if o.q<0] if electron else [o for o in self.quasiparticles if o.q>0]

        if detailed:
            for i in range(len(ehs)):
                o = ehs[i]

                Is = o.q * np.sum(o.vel * np.array((weightingFieldx_interp(o.pos),
                                weightingFieldy_interp(o.pos),
                                weightingFieldz_interp(o.pos))).T, axis=1)

                if len(Is) > 0:
                    func_I = interp1d(o.time, Is, bounds_error=False, fill_value=0)
                    induced_I += func_I(times_I)
        
        else:
            while len(ehs) > 0:
                o = ehs.pop()
                
                Is = o.q * np.sum(o.vel * np.array((weightingFieldx_interp(o.pos),
                            weightingFieldy_interp(o.pos),
                            weightingFieldz_interp(o.pos))).T, axis=1)

                if len(Is) > 0:
                    func_I = interp1d(o.time, Is, bounds_error=False, fill_value=0)
                    induced_I += func_I(times_I)
        
        dI = induced_I
        dT =times_I-start_time
        
        return dI, dT   
    
    def calculateIntegratedCharge(self, contact=0):
        try:
            self.dQ[contact] = cumulative_trapezoid(self.dI[contact], self.dt[contact], initial=0)
        except: 
            if not self.dI:
                print("No induced current to calculate charge from.")
        return None
    
    def sample(self, dt,length=None, contact=0):
        '''
        Samples the signal of an event to larger timesteps so that it is the same format as nab data. 
        There is some question as to the proper way to do this; for now we simply take the value at the exact time sampled.
        '''
        step = int(dt/(self.signal_times[contact][1]-self.signal_times[contact][0]))
        signal = self.signal[contact][::step].copy()
        if length==None:
            return signal
        else:
            signal = np.pad(signal, (round(length/2),0),"edge")
            if len(signal) >= length:
                return signal[:length]
            else:
                return np.pad(signal, (0,length-len(signal)), "edge")

def eventsFromG4root(filename, pixel=None, N=None, rotation = 0, nab_file = False):
    '''
    Imports event data from Geant4 root output.
    Assumes dE is in keV, pos is in mm, and time is in ns -> we convert to eV, m, s
    '''

    import uproot
    import pandas
    import awkward

    file = uproot.open(filename)

    angle_radians = np.radians(rotation)
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians), 0],
        [np.sin(angle_radians), np.cos(angle_radians), 0],
        [0, 0, 1]
    ])
    if nab_file:
        tree = file["dynamicTree"]
        keys = ["eventNum", "Hit_x", "Hit_y", "Hit_z", "Hit_time", "Hit_energy","Hit_particleType"]
        df = tree.arrays(keys, library="pd")

        flattened_dict = {}
        for key in keys:
            flattened_dict["%s"%key] = awkward.flatten(df["%s"%key])

        flattened_df = pandas.DataFrame(flattened_dict)
        gdf = flattened_df.groupby("eventNum")
        
        events = []
        i=0
        for eID, group in gdf:
            if N is not None and i > N:
                break
            particle_df = group.groupby("Hit_particleType")
            #TODO: standardize particle type tag
            for particleType, subgroup in particle_df:
                event = Event(eID, np.dot(subgroup[["Hit_x", "Hit_y", "Hit_z"]].to_numpy(), rotation_matrix.T),
                               subgroup["Hit_energy"].to_numpy(), subgroup["Hit_time"].to_numpy(),
                               subgroup["Hit_particleType"].to_numpy()[0])
                event.convertUnits(1e3,1e-3,1e-9)
                #TODO: ideally this should be done when we rotate the the coordinates initially but not sure how to 
                # cleanly grab the first position without looking it up...
                p0 = event.pos[0]
                if p0[2] < 0:
                    event.rotate_pos([0,180,0]) #TODO:not sure if I need to rotate about z too, center line of pixels are along beam, check that that is x axis
                    event.shift_pos(event.pos[0]-[0,0,1.2])
                    event.detector="Lower"
                else:
                    event.shift_pos(p0-[0,0,5])

                events.append(event)
            i+=1
    else:
        tree = file["ntuple/hits"]
        try:
            df = tree.arrays(["eventID", "trackID", "x", "y", "z", "time", "eDep", "pixelNumber", 'particle'], 
                            library="pd")
            if pixel is not None:
                df = df[df["pixelNumber"]==pixel]
        
            gdf = df.groupby("eventID")

            events = []
            i=0
            for eID, group in gdf:
                if N is not None and i > N:
                    break
                track_df = group.groupby("trackID")
                for trackID, subgroup in track_df:
                    event = Event(eID, np.dot(subgroup[["x", "y", "z"]].to_numpy(), rotation_matrix.T), subgroup["eDep"].to_numpy(),
                                subgroup["time"].to_numpy(), subgroup["particle"].to_numpy()[0])
                    event.convertUnits(1e3,1e-3,1e-9)
                    events.append(event)
                i+=1

        except KeyError:
            #old formatting exception
            df = tree.arrays(["iD", "x", "y", "z", "time", "eDep", "particle"], library="pd")
            gdf = df.groupby("iD")

            events = []
            i=0
            for eID, group in gdf:
                if N is not None and i > N:
                    break
                event = Event(eID, np.dot(group[["x", "y", "z"]].to_numpy(), rotation_matrix.T), group["eDep"].to_numpy(),
                            group["time"].to_numpy(), group["particle"].to_numpy()[0])
                event.convertUnits(1e3,1e-3,1e-9)
                events.append(event)
                i+=1

    return events

def save_to_deltaRice_dataset(events, waveform_length, filename, rice_parameter=8):
    import h5py
    import deltaRice.h5
    f = h5py.File(f'{filename}.h5', 'w')
    grp = f.create_group("waveforms")
    compression_opts = (rice_parameter, waveform_length)
    dtype='int16'
    chunk_size = 20 if 20 < len(events) else len(events)
    dataset = grp.create_dataset('singles', 
        (len(events), waveform_length), 
        compression=deltaRice.h5.H5FILTER,
        compression_opts = compression_opts,
        dtype=dtype, 
        chunks = (chunk_size, waveform_length))

    array = events.astype(dtype)

    dataset[:] = array

    f.close()

    return

def saveEventsNabPy(events, filename=None, dt=4e-9, length=7000, contact=0, hdf5=False):
    '''
    convert events to nabPy form, and save to pickle file or hdf5. 
    dt is time sampling in ns, which is 4ns for current nab daq.
    Saving to hdf5 uses delta rice compression. 
    '''
    new_events =np.array([event.sample(dt,length, contact) for event in events])

    if filename is not None and not hdf5:
        with open(filename+".pkl", "wb") as file:
            pickle.dump(new_events, file)

    if filename is not None and hdf5:
        save_to_deltaRice_dataset(new_events, length, filename)    
    
    return new_events
    
def loadEventsNabPy(filename):
    with open(filename, "rb") as file:
        events = pickle.load(file)
    
    return events
    
    
#save complete event object, for example if you want to be able to re-downsample, apply a different convolution, etc.
def saveEvents(events, filename):
    with open(filename+".pkl", "wb") as file:
        pickle.dump(events, file)
    return
    
def loadEvents(filename):
    with open(filename+".pkl", "rb") as file:
        events = pickle.load(file)
    return events
    
