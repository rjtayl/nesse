import numpy as np

class Potential:
    def __init__(self, _name, _data=None, _bounds=None):
        self.name = _name
        self.data = _data
        self.bounds = _bounds

    def __str__(self):
        return ("Nessie Potential object \n Name: %s \n Size: %s \n Bounds: %s \n" 
                % (self.name, np.shape(self.data), self.bounds))
            
class Field:
    def __init__(self, _name, _fieldx=None, _fieldy=None,_fieldz=None, _bounds=None):
        self.name = _name
        self.fieldx = _fieldx
        self.fieldy = _fieldy
        self.fieldz = _fieldz
        self.bounds = _bounds

    def __str__(self):
        return ("Nessie Field object \n Name: %s \n Size: %s \n Bounds: %s \n"
                % (self.name, np.shape(self.fieldx), self.bounds))

#Import weighting potential from hdf5 file.
def weightingPotentialFromH5(filename, rotate90=True):
    import h5py
    f = h5py.File(filename, 'r')
    data = np.array(f["wp"])
    if rotate90: data = np.rot90(data,axes=(0,2))
    boundsx = np.array(f["boundsx"])
    boundsy = np.array(f["boundsy"])
    boundsz = np.array(f["boundsz"])
    bounds = np.stack((boundsx,boundsy,boundsz))
    name = filename[:-3]
    weightingPotential = Potential(name, data, bounds)
    return weightingPotential

#Import electric field from hdf5 file.
def eFieldFromH5(filename, rotate90=True):
    import h5py
    f = h5py.File(filename, 'r')
    
    if rotate90:
        Ex = np.rot90(np.array(f["Ex"]),axes=(0,2))
        Ey = np.rot90(np.array(f["Ey"]),axes=(0,2))
        Ez = np.rot90(np.array(f["Ez"]),axes=(0,2))
    else:
        Ex = np.array(f["Ex"])
        Ey = np.array(f["Ey"])
        Ez = np.array(f["Ez"])
        
    boundsx = np.array(f["boundsx"])
    boundsy = np.array(f["boundsy"])
    boundsz = np.array(f["boundsz"])
    bounds = np.stack((boundsx,boundsy,boundsz))
    name = filename[:-3]

    Efield = Field(name, Ex, Ey,Ez, bounds)
    
    return Efield
