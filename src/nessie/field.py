import numpy as np
from scipy.interpolate import RegularGridInterpolator 

class Potential:
    def __init__(self, _name, _data=None, _grid=None):
        self.name = _name
        self.data = _data
        self.grid = _grid

    def __str__(self):
        return ("Nessie Potential object \n Name: %s \n Size: %s \n" 
                % (self.name, np.shape(self.data)))
            
class Field:
    def __init__(self, _name, _fieldx=None, _fieldy=None,_fieldz=None, _grid=None):
        self.name = _name
        self.fieldx = _fieldx
        self.fieldy = _fieldy
        self.fieldz = _fieldz
        self.grid = _grid

    def __str__(self):
        return ("Nessie Field object \n Name: %s \n Size: %s \n"
                % (self.name, np.shape(self.fieldx)))
                
    def interpolate(self):
        fieldShape = np.shape(self.fieldx)
        #x = np.linspace(fieldBounds[0][0],fieldBounds[0][1],fieldShape[0])
        #y = np.linspace(fieldBounds[1][0],fieldBounds[1][1],fieldShape[1])
        #z = np.linspace(fieldBounds[2][0],fieldBounds[2][1],fieldShape[2])
        
        x = self.grid[0]
        y = self.grid[1]
        z = self.grid[2]
        
        fieldx_interp  = RegularGridInterpolator((x,y,z),self.fieldx)
        fieldy_interp  = RegularGridInterpolator((x,y,z),self.fieldy)
        fieldz_interp  = RegularGridInterpolator((x,y,z),self.fieldz)
        
        fieldMag = np.sqrt(self.fieldx**2+self.fieldy**2+self.fieldz**2)
        fieldMag_interp = RegularGridInterpolator((x,y,z),fieldMag)
        
        return fieldx_interp, fieldy_interp, fieldz_interp, fieldMag_interp

#Import weighting potential from hdf5 file.
def weightingPotentialFromH5(filename, rotate90=True):
    import h5py
    f = h5py.File(filename, 'r')
    data = np.array(f["wp"])
    if rotate90: data = np.rot90(data,axes=(0,2))
    gridx = np.array(f["gridx"])
    gridy = np.array(f["gridy"])
    gridz = np.array(f["gridz"])
    grid = [gridx,gridy,gridz]
    name = filename[:-3]
    weightingPotential = Potential(name, data, grid)
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
        
    gridx = np.array(f["gridx"])
    gridy = np.array(f["gridy"])
    gridz = np.array(f["gridz"])
    grid = [gridx,gridy,gridz]
    name = filename[:-3]

    Efield = Field(name, Ex, Ey,Ez, grid)
    
    return Efield
