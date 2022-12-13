import numpy as np
from .constants import *
from scipy.interpolate import interp1d

class Event:
    def __init__(self, _id, _pos, _dE, _times):
        self.ID = _id
        self.pos = _pos
        self.dE = _dE
        self.times = _times

        self.dQ = []
        self.dI = []
        self.dt = []
        
        self.pos_drift_e = None
        self.times_drift_e = None
        
        self.pos_drift_h = None
        self.times_drift_h = None
        
        self.vel_drift_e = None
        self.vel_drift_h = None
        
        self.signal_I = None
        self.signal_times = None
        
    def convolveElectronicResponse(self, electronicResponse):
        func_I = interp1d(self.dt, self.dI, bounds_error=False, fill_value=0)
        temp_times = electronicResponse["times"]
        dt = np.diff(temp_times)[0]
        
        self.signal_I = np.convolve(func_I(temp_times),electronicResponse["step"])
        self.signal_times = np.arange(0,len(self.signal_I)*dt, dt)
        
        print(type(self.signal_I), type(self.signal_times))
        
        return None
        
    def setDriftPaths(self, pos, times, electron=True):
        if electron:
            self.pos_drift_e = pos
            self.times_drift_e = times
            print("Drift paths length: %d" % len(self.pos_drift_e))
        else:
            self.pos_drift_h = pos
            self.times_drift_h = times
        return None
        
    def convertUnits(self, energyConversionFactor, lengthConversionFactor, timeConversionFactor):
        self.dE *= energyConversionFactor
        self.pos *= lengthConversionFactor
        self.times *= timeConversionFactor
        return None
        
    def getDriftVelocities(self):
        vel_drift_e = []
        vel_drift_h = []
        for i in range(len(self.pos_drift_e)):
            pos = self.pos_drift_e[i]
            pos_e = [pos[:,0], pos[:,1], pos[:,2]]
            dx,dy,dz = np.diff(pos_e)/np.diff(self.times_drift_e[i])
            vel_drift_e.append(np.stack((dx,dy,dz),axis=-1))
        
        for i in range(len(self.pos_drift_h)):
            pos = self.pos_drift_h[i]
            pos_h = [pos[:,0], pos[:,1], pos[:,2]]
            dx,dy,dz = np.diff(pos_h)/np.diff(self.times_drift_h[i])
            vel_drift_h.append(np.stack((dx,dy,dz),axis=-1))
            
        self.vel_drift_e = vel_drift_e
        self.vel_drift_h = vel_drift_h
        return None
        
    def calculateInducedCurrent(self,weightingField, dt, interp3d=False):
        weightingFieldx_interp, weightingFieldy_interp, weightingFieldz_interp, weightingFieldMag_interp = weightingField.interpolate(interp3d=interp3d)
        
        max_time = max([max([times[-1] for times in self.times_drift_e]),max([times[-1] for times in self.times_drift_h])])
        times_I = np.arange(0,max_time,dt)
        
        induced_I = np.zeros(len(times_I))

        for i in range(len(self.vel_drift_e)):
            if interp3d:
                Is = [-qe_SI*np.dot(self.vel_drift_e[i][j], 
                        [weightingFieldx_interp(self.pos_drift_e[i][j]),
                         weightingFieldy_interp(self.pos_drift_e[i][j]),
                         weightingFieldz_interp(self.pos_drift_e[i][j])]) for j in range(len(self.vel_drift_e[i]))]
                         
            else:
                Is = [-qe_SI*np.dot(self.vel_drift_e[i][j], 
                        [weightingFieldx_interp(self.pos_drift_e[i][j]),
                         weightingFieldy_interp(self.pos_drift_e[i][j]),
                         weightingFieldz_interp(self.pos_drift_e[i][j])])[0] for j in range(len(self.vel_drift_e[i]))]

            times = self.times_drift_e[i][:-1]

            if len(Is)>1:
                func_I = interp1d(times, Is, bounds_error=False, fill_value=0)
                induced_I += func_I(times_I)
                
                
        #times = self.times_drift_e[i][:-1] + self.times[i]
        for i in range(len(self.vel_drift_h)):
            if interp3d:
                Is = [qe_SI*np.dot(self.vel_drift_h[i][j], 
                        [weightingFieldx_interp(self.pos_drift_h[i][j]),
                         weightingFieldy_interp(self.pos_drift_h[i][j]),
                         weightingFieldz_interp(self.pos_drift_h[i][j])]) for j in range(len(self.vel_drift_h[i]))]
                         
            else:
                Is = [qe_SI*np.dot(self.vel_drift_h[i][j], 
                        [weightingFieldx_interp(self.pos_drift_h[i][j]),
                         weightingFieldy_interp(self.pos_drift_h[i][j]),
                         weightingFieldz_interp(self.pos_drift_h[i][j])])[0] for j in range(len(self.vel_drift_h[i]))]
            times = self.times_drift_h[i][:-1]

            if len(Is)>1:
                func_I = interp1d(times, Is, bounds_error=False, fill_value=0)
                induced_I += func_I(times_I)
            
        #times = self.times_drift_h[i][:-1] + self.times[i]
        
        self.dI = induced_I
        self.dt = times_I
        
        return None           
            

def eventsFromG4root(filename):
    import uproot
    import pandas

    file = uproot.open(filename)
    tree = file["ntuple/hits"]

    try:
        df = tree.arrays(["eventID", "trackID", "x", "y", "z", "time", "eDep"], 
                         library="pd")
        gdf = df.groupby("eventID")
    except:
        #old formatting exception
        df = tree.arrays(["iD", "x", "y", "z", "time", "eDep"], library="pd")
        gdf = df.groupby("iD")

    events = []
    for eID, group in gdf:
        event = Event(eID, group[["x", "y", "z"]].to_numpy(), group["eDep"].to_numpy(), group["time"].to_numpy())
        event.convertUnits(1e3,1e-3,1e-9)
        events.append(event)
    return events
    

