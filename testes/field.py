import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib
import matplotlib.animation as ani
import time
from enum import Enum

class GridSet(Enum):
    BASE = 0
    SUPPORT = 1

class DiscreteVectorField:
    
    
    def __init__(self, side_length: int, grid_set : GridSet):
        self.side_length = side_length

        if grid_set == GridSet.BASE:
            fx = np.zeros([side_length, side_length, side_length]) 
            fy = np.zeros([side_length, side_length, side_length]) 
            fz = np.zeros([side_length, side_length, side_length]) 
        elif grid_set == GridSet.SUPPORT:
            fx = np.zeros([side_length-1, side_length-1, side_length]) 
            fy = np.zeros([side_length-1, side_length-1, side_length]) 
            fz = np.zeros([side_length, side_length, side_length]) 
        self.field = np.stack((fx, fy, fz), axis=-1)

    def compute_curl(self):
        fx = self.field[:,:,:,0]
        fy = self.field[:,:,:,1]
        fz = self.field[:,:,:,2]

        self.pypx = np.diff(fy, axis=0)[:, 0:-1,:]
        self.pzpx = np.diff(fz, axis=0)[:, :, 0:-1]

        self.pxpy = np.diff(fx, axis=1)[0:-1, :, :]
        self.pzpy = np.diff(fz, axis=1)[:, :, 0:-1]

        self.pxpz = np.diff(fx, axis=2)[0:-1, :, :]
        self.pypz = np.diff(fy, axis=2)[:, 0:-1, :]
        
        x_comp = self.pzpy - self.pypz
        y_comp = self.pxpz - self.pzpx
        z_comp = self.pypx - self.pxpy
        
        return x_comp, y_comp, z_comp

# CONSTANTS

eps0 = 8.8541878188e-12
mu0 = (4 * np.pi) * 1e-7
c0 = 1 / np.sqrt(mu0 * eps0)
imp0 = np.sqrt(mu0 / eps0)

# CONSTANT PARAMETERS

FPS = 60
SPACE_SIDE_LENGTH = 600 # m
MAXIMUM_FREQUENCE = 5e6 # 1 Mhz
SIMULATION_TIME = 10
TIME_SCALE = 1e-6

# DERIVED PARAMETERS
frame_time = 1/FPS 
ds = c0 / (MAXIMUM_FREQUENCE * 10)
dt = ds/(2 * c0)
jmax = int(SPACE_SIDE_LENGTH  / ds)

e_field = DiscreteVectorField(jmax)
h_field = DiscreteVectorField(jmax-1)

# PLOTS
jsource = int(jmax/2)
fig = plt.figure()
ax = fig.add_subplot()
Explot = ax.pcolormesh(e_field.field[:, jsource, :, 0])
Explot.set_norm(matplotlib.colors.Normalize(vmin=-3, vmax=3))


## SOURCE


def source_function(t):
    f0 = 3e6
    w0 = 2 * np.pi * f0
    tau = 1/5
    return np.exp(-((t/TIME_SCALE - 1)**2/tau**2)) * np.sin(w0*t)

# SIMULATION

simulation_steps_per_frame = int((frame_time * TIME_SCALE) / dt)
max_frames = int(SIMULATION_TIME / (simulation_steps_per_frame * dt))

current_n = 0
def update(_):
    global h_field, e_field, current_n
    for n in range(current_n, current_n+simulation_steps_per_frame):
        hrx, hry, hrz = h_field.compute_curl()
        e_field.field[0:-1, 1:-1, 1:-1, 0] += 0.5 * hrx
        e_field.field[1:-1, 0:-1, 1:-1, 1] += 0.5 * hry
        e_field.field[1:-1, 1:-1, 0:-1, 2] += 0.5 * hrz

        e_field.field[jsource, jsource, jsource, 2] = source_function(n*dt)

        erx, ery, erz = e_field.compute_curl()
        h_field.field[:, :, :, 0] -= 0.5 * erx[0:-1, :, :]
        h_field.field[:, :, :, 1] -= 0.5 * ery[:, 0:-1, :]
        h_field.field[:, :, :, 2] -= 0.5 * erz[:, :, 0:-1]

    current_n += simulation_steps_per_frame
    Explot.set_array(e_field.field[:, :, jsource, 2])
    ax.set_title(int(current_n * dt / TIME_SCALE))

animation = ani.FuncAnimation(fig=fig, func=update, frames=max_frames, interval=1000/FPS)
plt.show()