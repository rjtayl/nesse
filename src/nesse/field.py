from scipy.interpolate import RegularGridInterpolator 
from .interp_3d import *
import numpy as np
import h5py

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
    
    def shift(self, shift=(0,0,0)):
        for i in [0,1,2]:
            self.grid[i] = self.grid[i]+shift[i]

    def toField(self):
        gradient = np.gradient(self.data, self.grid[0],self.grid[1], self.grid[2])
        return Field("Weighting Field", gradient[0],gradient[1],gradient[2],
                                          self.grid)
    
    def saveToH5(self, filename):
        if filename is None: filename = self.name
        f_new = h5py.File(filename + ".hf", "w")
        f_new["wp"] = self.data
        f_new['gridx'] = self.grid[0]
        f_new['gridy'] = self.grid[1]
        f_new['gridz'] = self.grid[2]
        f_new.close()

    def interpolate(self, interp3d=False):
        '''
        Returns linear interpolations of the field. 
        interp3d is a cython implementation of grid interpolation, we have modified this from https://github.com/jglaser/interp3d to work for non-regular grids. 
        If interp3d is set to False instead the scipy RegularGridInterpolator will be used. 
        This is a pure python interpolator and is significantly slower. 
        Each interpolator takes input differently so which is used is tracked throughout the simulation.  
        '''
        x = self.grid[0]
        y = self.grid[1]
        z = self.grid[2]
        
        if interp3d:
            
            interp = Interp3D(self.data.astype('double', order="C"), x,y,z)
            
        else:   
            interp = RegularGridInterpolator((x,y,z),self.data)
            
        return interp

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
    
    def saveToH5(self, filename=None):
        if filename is None: filename = self.name
        f_new = h5py.File(filename + ".hf", "w")
        f_new["Ex"] = self.fieldx
        f_new["Ey"] = self.fieldy
        f_new["Ez"] = self.fieldz
        f_new['gridx'] = self.grid[0]
        f_new['gridy'] = self.grid[1]
        f_new['gridz'] = self.grid[2]
        f_new.close()
 

def potentialFromH5(filename, rotate90=False):
    '''
    Import potential from hdf5 file. 
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
    potential = Potential(name, data, grid)
    return potential

#Import field from hdf5 file.
def fieldFromH5(filename, rotate90=False):
    '''
    Import field from hdf5 file. 
    When saving files to hdf5 from Julia a rotation is applied to the array, so when importing from SSD we need to rotate back.
    '''
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

    field = Field(name, Ex, Ey,Ez, grid)
    
    return field

def analytical_potential(name, P, bounds,dx,dy,dz):
    '''
    Create a potential using an analytical function of the form lambda x,y,x: f(x,y,z).
    You can then convert this to a field in the normal way to get analytical fields. 
    '''
    x = np.arange(bounds[0][0],bounds[0][1],dx)
    y = np.arange(bounds[1][0],bounds[1][1],dy)
    z = np.arange(bounds[2][0],bounds[2][1],dz)
    grid = [x, y, z]
    X,Y,Z = np.meshgrid(x,y,z,indexing='xy')
    data = P(X,Y,Z)
    return Potential(name, data, grid)

def analytical_field(name, fields, bounds,dx,dy,dz):
    '''
    Create a potential using an analytical function of the form lambda x,y,x: f(x,y,z).
    You can then convert this to a field in the normal way to get analytical fields. 
    '''
    x = np.arange(bounds[0][0],bounds[0][1],dx)
    y = np.arange(bounds[1][0],bounds[1][1],dy)
    z = np.arange(bounds[2][0],bounds[2][1],dz)
    grid = [x, y, z]
    X,Y,Z = np.meshgrid(x,y,z,indexing='xy')
    Ex = fields[0](X,Y,Z)
    Ey = fields[1](X,Y,Z)
    Ez = fields[2](X,Y,Z)
    return Field(name, Ex, Ey, Ez, grid)
