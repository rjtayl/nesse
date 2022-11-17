class Event:
    def __init__(self, _id, _pos, _dE, _t0):
        self.ID = _id
        self.pos = _pos
        self.dE = _dE
        self.t0 = _t0

        self.dQ = []
        self.dI = []

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
        events.append(event)
    return events
