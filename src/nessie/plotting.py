import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.animation as animation

plt.rcParams['figure.figsize'] = [8, 5]
plt.rcParams.update({'font.size': 18})

def plot_event_drift(event, bounds, prefix="",suffix=""):
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
    plt.show()
    
    plt.savefig(prefix+"drift_path" +suffix + ".png")
    return None
