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
    
    
    def __init__(self, side_length):
        self.side_length = side_length
        fx = np.zeros(side_length) 
        fy = np.zeros(side_length) 
        fz = np.zeros(side_length) 
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

WAVE_GUIDE_Y = 0.1
WAVE_GUIDE_Z = 0.1


FPS = 60
SPACE_SIDE_LENGTH_X = 0.9 # m
SPACE_SIDE_LENGTH_Y = 0.45 # m
SPACE_SIDE_LENGTH_Z = 0.45 # m
MAXIMUM_FREQUENCE = 5e9 # 1 Ghz
SIMULATION_TIME = 10
TIME_SCALE = 1e-9

# DERIVED PARAMETERS
frame_time = 1/FPS 
ds = c0 / (MAXIMUM_FREQUENCE * 10)
dt = ds/(2 * c0)
jmaxs = np.array([int(SPACE_SIDE_LENGTH_X  / ds), int(SPACE_SIDE_LENGTH_Y  / ds), int(SPACE_SIDE_LENGTH_Z  / ds)])

e_field = DiscreteVectorField(jmaxs)
h_field = DiscreteVectorField(jmaxs)



Xz,Zx = np.meshgrid(np.arange(jmaxs[0]), np.arange(jmaxs[2]))
Yz,Zy = np.meshgrid(np.arange(jmaxs[1]), np.arange(jmaxs[2]))
# PLOTS

ref = (jmaxs/2).astype(np.int32)
ref[0] -= 3


fig = plt.figure(figsize=[10,4])
fig.suptitle("E Field Magnitude: 0 ns")

ax1, ax2 = fig.subplots(1, 2)
ax1.set_aspect(1.3*jmaxs[2]/jmaxs[0])
ax2.set_aspect(jmaxs[2]/jmaxs[1])
Explot = ax1.pcolormesh(Xz*ds*100,Zx*ds*100, e_field.field[:, ref[1], :, 0].T)
Explot2 = ax2.pcolormesh(Yz*ds*100,Zy*ds*100, e_field.field[ref[0], :, :, 0].T)
Explot.set_norm(matplotlib.colors.Normalize(vmin=0, vmax=0.0001))
Explot2.set_norm(matplotlib.colors.Normalize(vmin=0, vmax=0.001))
ax1.set_xlabel("X (cm)")
ax1.set_ylabel("Z (cm)")
ax2.set_xlabel("Y (cm)")
ax2.set_ylabel("Z (cm)")

## SOURCE

w_y = int(WAVE_GUIDE_Y/ds)
w_z = int(WAVE_GUIDE_Z/ds)
w_y, w_y_end = int((jmaxs[1]-w_y)/2), int((jmaxs[1]+w_y)/2)
w_z, w_z_end = jmaxs[2]-w_z, jmaxs[2]
jsource=(jmaxs[0]-3, int(w_y/2 + w_y_end/2), int(w_z/2 + w_z_end/2))

microwave = np.ones([jmaxs[1], jmaxs[2]])
microwave[1:, 1:] = 0
microwave[w_y:w_y_end, w_z:w_z_end] = 1
microwave = np.repeat(microwave[np.newaxis, :, :], jmaxs[0], axis=0)
microwave[:int(jmaxs[0]/2), :, :] = 1


#microwave[3:int(jmax/2),3:int(jmax/2), int(jmax/2)-1] = 0
#microwavePlot = ax1.pcolormesh(Xz*ds*100,Zx*ds*100, microwave[:, jsource[1]].T)
#microwavePlot2 = ax2.pcolormesh(Yz*ds*100,Zy*ds*100, microwave[jsource[0], :].T)
microwave = microwave[:, :, :, np.newaxis]

def source_function(t):
    f0 = 5e9
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
        e_field.field *= microwave
        e_field.field[jsource[0],jsource[1], jsource[2], 0] = source_function(n*dt)

        h_field.field[1:, 1:-1, 1:-1, 0] += 0.5 * (e_field.field[1:, 1:-1, 2:, 1] - e_field.field[1:, 1:-1, 1:-1, 1] - e_field.field[1:, 2:, 1:-1, 2] + e_field.field[1:, 1:-1, 1:-1, 2])
        h_field.field[1:-1, 1:, 1:-1, 1] += 0.5 * (e_field.field[2:, 1:, 1:-1, 2] - e_field.field[1:-1, 1:, 1:-1, 2] - e_field.field[1:-1, 1:, 2:, 0] + e_field.field[1:-1, 1:, 1:-1, 0])
        h_field.field[1:-1, 1:-1, 1:, 2] += 0.5 * (e_field.field[1:-1, 2:, 1:, 0] - e_field.field[1:-1, 1:-1, 1:, 0] - e_field.field[2:, 1:-1, 1:, 1] + e_field.field[1:-1, 1:-1, 1:, 1])


    print(f"{format(np.round(current_n * dt / TIME_SCALE, 2), ".2f")} ns")
    current_n += simulation_steps_per_frame
    pont = np.linalg.cross(e_field.field, h_field.field)
    mag = (np.sqrt(np.sum(np.pow(pont,2), axis=-1)))
    Explot.set_array(mag[:, ref[1], :].T)
    Explot2.set_array(mag[ref[0], :, :].T)
    fig.suptitle(f"E Field Magnitude: {format(np.round(current_n * dt / TIME_SCALE, 2), ".2f")} ns")

animation = ani.FuncAnimation(fig=fig, func=update, frames=max_frames, interval=1000/FPS)
#animation.save("simulation.mp4", writer="ffmpeg", fps=FPS, dpi=150)
plt.show()