class Event:
    def __init__(self, _id, _pos, _dE, _t0):
        ID = _id
        pos = _pos
        dE = _dE
        t0 = _t0

        dQ = []
        dI = []

def eventsFromG4root(filename):
    import uproot
    import pandas

    file = uproot.open(filename)
    tree = file["ntuple/hits"]

    df = tree.arrays(["eventID", "trackID", "x", "y", "z", "time", "eDep"], library="pd")
    gdf = df.groupby("eventID")

    events = []
    for eID, group in gdf:
        event = Event(eID, group[["x", "y", "z"]].to_numpy(), group["eDep"].to_numpy(), group["time"].to_numpy())
        events.append(event)
    return events
