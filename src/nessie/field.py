from scipy.interpolate import RegularGridInterpolator 
from .interp_3d import *
import numpy as np

class Potential:
    '''
    The Potential class is used to create electric and weighting potential objects. 
    These can then be converted to Field objects for simulation. 
    We use Potential objects mainly for plotting and transferring field information between nessie and our other field solving tools. 
    data is assumed to be a 3-dimensional array. grid is an array of axes (x,y,z) where each axis is an array of positions in meters. 
    '''
    def __init__(self, _name, _data=None, _grid=None):
        self.name = _name
        self.data = _data
        self.grid = _grid

    def __str__(self):
        return ("Nessie Potential object \n Name: %s \n Size: %s \n" 
                % (self.name, np.shape(self.data)))

class Field:
    '''
    The Field class is used to create electric and weighting field objects. 
    We use cartesian coordinates and define each field component as its own 3-dimensional array (fieldx, fieldy, fieldz). 
    grid is the same as in the Potential class. 
    '''
    
    def __init__(self, _name, _fieldx=None, _fieldy=None,_fieldz=None, _grid=None):
        self.name = _name
        self.fieldx = _fieldx
        self.fieldy = _fieldy
        self.fieldz = _fieldz
        self.grid = _grid

    def __str__(self):
        return ("Nessie Field object \n Name: %s \n Size: %s \n"
                % (self.name, np.shape(self.fieldx)))
     
    def interpolate(self, interp3d=False):
        '''
        Returns linear interpolations of the field. 
        interp3d is a cython implementation of grid interpolation, we have modified this from https://github.com/jglaser/interp3d to work for non-regular grids. 
        If interp3d is set to False instead the scipy RegularGridInterpolator will be used. 
        This is a pure python interpolator and is significantly slower. 
        Each interpolator takes input differently so which is used is tracked throughout the simulation.  
        '''
        fieldShape = np.shape(self.fieldx)
        x = self.grid[0]
        y = self.grid[1]
        z = self.grid[2]
        
        if interp3d:
            
            fieldx_interp = Interp3D(self.fieldx.astype('double', order="C"), x,y,z)
            fieldy_interp = Interp3D(self.fieldy.astype('double',order="C"), x,y,z)
            fieldz_interp = Interp3D(self.fieldz.astype('double',order="C"), x,y,z)
            
            fieldMag = np.sqrt(self.fieldx**2+self.fieldy**2+self.fieldz**2)
            fieldMag_interp = Interp3D(fieldMag.astype('double',order="C"), x,y,z)
            
        else:   
            fieldx_interp = RegularGridInterpolator((x,y,z),self.fieldx)
            fieldy_interp = RegularGridInterpolator((x,y,z),self.fieldy)
            fieldz_interp = RegularGridInterpolator((x,y,z),self.fieldz)
            fieldMag = np.sqrt(self.fieldx**2+self.fieldy**2+self.fieldz**2)
            fieldMag_interp = RegularGridInterpolator((x,y,z),fieldMag)
        
        return fieldx_interp, fieldy_interp, fieldz_interp, fieldMag_interp


def weightingPotentialFromH5(filename, rotate90=True):
    '''
    Import weighting potential from hdf5 file. 
    When saving files to hdf5 from Julia a rotation is applied to the array, so when importing from SSD we need to rotate back.
    '''
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
