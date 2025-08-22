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
    
    
    def __init__(self, side_length: int):
        self.side_length = side_length
        fx = np.zeros([side_length, side_length, side_length]) 
        fy = np.zeros([side_length, side_length, side_length]) 
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
SPACE_SIDE_LENGTH = 0.6 # m
MAXIMUM_FREQUENCE = 5e9 # 1 Ghz
SIMULATION_TIME = 10
TIME_SCALE = 1e-9

# DERIVED PARAMETERS
frame_time = 1/FPS 
ds = c0 / (MAXIMUM_FREQUENCE * 10)
dt = ds/(2 * c0)
jmax = int(SPACE_SIDE_LENGTH  / ds)

e_field = DiscreteVectorField(jmax)
h_field = DiscreteVectorField(jmax)

# PLOTS
jsource = int(jmax/2)
fig = plt.figure()
ax = fig.add_subplot()
Explot = ax.pcolormesh(e_field.field[:, jsource, :, 0])
Explot.set_norm(matplotlib.colors.Normalize(vmin=-0.001, vmax=0.001))


## SOURCE


def source_function(t):
    f0 = .5e9
    w0 = 2 * np.pi * f0
    return np.sin(w0*t)

# SIMULATION

simulation_steps_per_frame = int((frame_time * TIME_SCALE) / dt)
max_frames = int(SIMULATION_TIME / (simulation_steps_per_frame * dt))

current_n = 0
def update(_):
    global h_field, e_field, current_n
    for n in range(current_n, current_n+simulation_steps_per_frame):

        e_field.field[1:, 1:, 1:, 0] += 0.5 * (h_field.field[1:, 1:, 1:, 2] - h_field.field[1:, 0:-1, 1:, 2] - h_field.field[1:, 1:, 1:, 1] + h_field.field[1:, 1:, 0:-1, 1])
        e_field.field[1:, 1:, 1:, 1] += 0.5 * (h_field.field[1:, 1:, 1:, 0] - h_field.field[1:, 1:, 0:-1, 0] - h_field.field[1:, 1:, 1:, 2] + h_field.field[0:-1, 1:, 1:, 2])
        e_field.field[1:, 1:, 1:, 2] += 0.5 * (h_field.field[1:, 1:, 1:, 1] - h_field.field[0:-1, 1:, 1:, 1] - h_field.field[1:, 1:, 1:, 0] + h_field.field[1:, 0:-1, 1:, 0])

        e_field.field[jsource, jsource, jsource, 2] = source_function(n*dt)

        h_field.field[1:, 1:-1, 1:-1, 0] += 0.5 * (e_field.field[1:, 1:-1, 2:, 1] - e_field.field[1:, 1:-1, 1:-1, 1] - e_field.field[1:, 2:, 1:-1, 2] + e_field.field[1:, 1:-1, 1:-1, 2])
        h_field.field[1:-1, 1:, 1:-1, 1] += 0.5 * (e_field.field[2:, 1:, 1:-1, 2] - e_field.field[1:-1, 1:, 1:-1, 2] - e_field.field[1:-1, 1:, 2:, 0] + e_field.field[1:-1, 1:, 1:-1, 0])
        h_field.field[1:-1, 1:-1, 1:, 2] += 0.5 * (e_field.field[1:-1, 2:, 1:, 0] - e_field.field[1:-1, 1:-1, 1:, 0] - e_field.field[2:, 1:-1, 1:, 1] + e_field.field[1:-1, 1:-1, 1:, 1])


    current_n += simulation_steps_per_frame
    mag = (np.sqrt(np.sum(np.pow(e_field.field,2), axis=-1)))
    Explot.set_array(mag[jsource, :, :])
    ax.set_title(int(current_n * dt / TIME_SCALE))

animation = ani.FuncAnimation(fig=fig, func=update, frames=max_frames, interval=1000/FPS)
plt.show()