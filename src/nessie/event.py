class Event:
    def __init__(self, _id, _pos, _dE, _times):
        self.ID = _id
        self.pos = _pos
        self.dE = _dE
        self.times = _times

        self.dQ = []
        self.dI = []
        
        self.pos_drift=[]
        self.times_drift=[]
        
    def setDriftPaths(self, pos, times):
        self.pos_drift = pos
        self.times_drift=times
        return None
        
    def convertUnits(self, lengthConversionFactor, timeConversionFactor):
        self.pos = self.pos*lengthConversionFactor
        self.times = self.times*timeConversionFactor
        return None

def eventsFromG4root(filename):
    import uproot
    import pandas

    file = uproot.open(filename)
    tree = file["ntuple/hits"]

    try:
        df = tree.arrays(["eventID", "trackID", "x", "y", "z", "time", "eDep"], library="pd")
        gdf = df.groupby("eventID")
    except:
        #old formatting exception
        df = tree.arrays(["iD", "x", "y", "z", "time", "eDep"], library="pd")
        gdf = df.groupby("iD")

    events = []
    for eID, group in gdf:
        event = Event(eID, group[["x", "y", "z"]].to_numpy(), group["eDep"].to_numpy(), group["time"].to_numpy())
        event.convertUnits(1e-3,1)
        events.append(event)
    return events
    

