import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.animation as animation
import numpy as np
from scipy.interpolate import RegularGridInterpolator 

plt.rcParams['figure.figsize'] = [8, 5]
plt.rcParams.update({'font.size': 18})

def plot_event_drift(event, bounds, prefix="",suffix="", show_plot=True):
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    
    for i in range(len(event.pos_drift)):
        x = event.pos_drift[i][:,0]
        y = event.pos_drift[i][:,1]
        z = event.pos_drift[i][:,2]
        ax.plot3D(x, y, z, 'green')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(*bounds[0])
    ax.set_ylim(*bounds[1])
    ax.set_zlim(*bounds[2])
    
    if show_plot: plt.show()
    
    plt.savefig(prefix+"drift_path" +suffix + ".png")
    return None
    
def plot_field_lines(field, bounds, mesh_size = (500,500), x_plane = True, density = 2,
                        prefix="",suffix="", show_plot=True):
    fieldBounds = field.bounds
    fieldShape = np.shape(field.fieldx)
    x = np.linspace(fieldBounds[0][0],fieldBounds[0][1],fieldShape[0])
    y = np.linspace(fieldBounds[1][0],fieldBounds[1][1],fieldShape[1])
    z = np.linspace(fieldBounds[2][0],fieldBounds[2][1],fieldShape[2])
        
    fieldx_interp  = RegularGridInterpolator((x,y,z),field.fieldx)
    fieldy_interp  = RegularGridInterpolator((x,y,z),field.fieldy)
    fieldz_interp  = RegularGridInterpolator((x,y,z),field.fieldz)
    
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
        Ex, Ez = fieldy_interp(coords), fieldz_interp(coords)
    else:
        #E = lambda x,z:fieldy_interp([x,0,z]),fieldz_interp([x,0,z])
        i = np.linspace(bounds[0][0], bounds[0][1], ni)
        j = np.linspace(bounds[2][0], bounds[2][1], nj)
        X, Z = np.meshgrid(i,j)
        Y = np.zeros((ni,nj))
        coords = np.stack((X,Y,Z),axis=-1)
        Ex, Ez = fieldy_interp(coords), fieldz_interp(coords)
    
    
    
    
    
    color = 2 * np.log(np.hypot(Ex, Ez))
    
    fig, ax = plt.subplots()
    
    strm = ax.streamplot(i, j, Ex, Ez, color=color, linewidth=1, cmap=plt.cm.inferno,
              density=density, arrowstyle='->', arrowsize=1.5)
    fig.colorbar(strm.lines)
              
    xlabel = "y" if x_plane else "x"          
    ax.set_xlabel(xlabel) 
    ax.set_ylabel('z')
    ax.set_xlim(i[0],i[-1])
    ax.set_ylim(j[0],j[-1])
    
    if show_plot: plt.show()
    
    plt.savefig(prefix+"field" +suffix + ".png")
    
    return None
