import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.animation as animation
import numpy as np
from scipy.interpolate import RegularGridInterpolator 

plt.rcParams['figure.figsize'] = [8, 5]
plt.rcParams.update({'font.size': 18})

def plot_signal(event, show_plot=True, alpha=1):
    plt.plot(event.dt,event.dI, alpha=alpha)
    if show_plot: plt.show()
    return None

def plot_event_drift(event, bounds, prefix="",suffix="", show_plot=True):
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    
    try:
	    for i in range(len(event.pos_drift_e)):
	        x = event.pos_drift_e[i][:,0]
	        y = event.pos_drift_e[i][:,1]
	        z = event.pos_drift_e[i][:,2]
	        ax.plot3D(x, y, z, 'green')
    except: print("No electron drift paths")
	
    try:
        for i in range(len(event.pos_drift_h)):
            x = event.pos_drift_h[i][:,0]
            y = event.pos_drift_h[i][:,1]
            z = event.pos_drift_h[i][:,2]
            ax.plot3D(x, y, z, 'red')
    except: print("No hole drift paths")
	
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(*bounds[0])
    ax.set_ylim(*bounds[1])
    ax.set_zlim(*bounds[2])
    
    if show_plot: plt.show()
    
    plt.savefig(prefix+"drift_path" +suffix + ".png")
    return None
    
def plot_field_lines(field, mesh_size = (500,500), x_plane = True, density = 2,
                        prefix="",suffix="", show_plot=True, log=True, bounds=None):
                        
    fieldx_interp, fieldy_interp, fieldz_interp, fieldMag_interp = field.interpolate()
    
    if bounds is None:
        bounds = [[axis[0],axis[-1]] for axis in field.grid]
    
    ni, nj = mesh_size[0], mesh_size[1]
    
    if x_plane:
        #E = lambda y,z:fieldy_interp([0,y,z]),fieldz_interp([0,y,z])
        i = np.linspace(bounds[1][0], bounds[1][1], ni)
        j = np.linspace(bounds[2][0], bounds[2][1], nj)
        Y, Z = np.meshgrid(i,j)
        X = np.zeros((ni,nj))
        #print(X,Z)
        #print(np.shape(X), np.shape(Z))
        #print(fieldy_interp([0,X,Z]))
        coords = np.stack((X,Y,Z),axis=-1)
        Ex, Ey, Ez = fieldy_interp(coords), fieldz_interp(coords), fieldz_interp(coords)
    else:
        #E = lambda x,z:fieldy_interp([x,0,z]),fieldz_interp([x,0,z])
        i = np.linspace(bounds[0][0], bounds[0][1], ni)
        j = np.linspace(bounds[2][0], bounds[2][1], nj)
        X, Z = np.meshgrid(i,j)
        Y = np.zeros((ni,nj))
        coords = np.stack((X,Y,Z),axis=-1)
        Ex, Ey, Ez = fieldy_interp(coords),fieldy_interp(coords), fieldz_interp(coords)
    
    color = np.sqrt(Ex**2+Ey**2+Ez**2) if log else Ez
    
    fig, ax = plt.subplots()
    
    strm = ax.streamplot(i, j, Ex, Ez, color=color, linewidth=1, cmap=plt.cm.inferno,
              density=density, arrowstyle='->', arrowsize=1.5)
    fig.colorbar(strm.lines, label="Electric Field Magnitude (V/m)")
              
    xlabel = "y" if x_plane else "x"          
    ax.set_xlabel(xlabel) 
    ax.set_ylabel('z')
    ax.set_xlim(i[0],i[-1])
    ax.set_ylim(j[0],j[-1])
    
    if show_plot: plt.show()
    
    plt.savefig(prefix+"field" +suffix + ".png")
    
    return None
    
def plot_potential(potential, mesh_size = (500,500), x_plane = True,
                        prefix="",suffix="", show_plot=True, bounds=None):
                            
    x = potential.grid[0]
    y = potential.grid[1]
    z = potential.grid[2]
    
    if bounds is None:
        bounds = [[axis[0],axis[-1]] for axis in potential.grid]
           
    potential_interp  = RegularGridInterpolator((x,y,z),potential.data)
    
    ni, nj = mesh_size[0], mesh_size[1]
    
    if x_plane:
        i = np.linspace(bounds[1][0], bounds[1][1], ni)
        j = np.linspace(bounds[2][0], bounds[2][1], nj)
        Y, Z = np.meshgrid(i,j)
        X = np.zeros((ni,nj))
        coords = np.stack((X,Y,Z),axis=-1)
        potentialGrid = potential_interp(coords)
        plt.contourf(Y,Z,potentialGrid,mesh_size[0])
    else:
        i = np.linspace(bounds[0][0], bounds[0][1], ni)
        j = np.linspace(bounds[2][0], bounds[2][1], nj)
        X, Z = np.meshgrid(i,j)
        Y = np.zeros((ni,nj))
        coords = np.stack((X,Y,Z),axis=-1)
        potentialGrid = potential_interp(coords)
        plt.contourf(X,Z,potentialGrid,mesh_size[0])
    
    if show_plot: plt.show()
    
    plt.savefig(prefix+"potential" +suffix + ".png")
    
    return None
