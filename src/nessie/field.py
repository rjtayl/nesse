import numpy as np

class Field:
    def __init__(self, _name, _data, _bounds):
        self.name = _name
        self.data = _data
        self.bounds = _bounds

    def __str__(self):
        return ("Nessie Field Object \n Name: %s \n Size: %s \n"
                % (self.name, np.shape(self.data)))

#Import weighting potential from hdf5 file.
def wpFromH5(filename, SSD=True):
    import h5py
    f = h5py.File(filename, 'r')
    data = np.array(f["wp"])
    if SSD: data = np.rot90(data,axes=(0,2))
    boundsx = np.array(f["boundsx"])
    boundsy = np.array(f["boundsy"])
    boundsz = np.array(f["boundsz"])
    bounds = np.stack((boundsx,boundsy,boundsz))
    name = filename[:-3]
    field = Field(name, data, bounds)
    return field

#Import electric field from hdf5 file.
def efFromH5(filename, SSD=True):
    import h5py
    f = h5py.File(filename, 'r')
    
    if SSD:
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

    fieldx = Field(name+'_Ex', Ex, bounds)
    fieldy = Field(name+'_Ey', Ey, bounds)
    fieldz = Field(name+'_Ez', Ez, bounds)
    
    return fieldx, fieldy, fieldy
