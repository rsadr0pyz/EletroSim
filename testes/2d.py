import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from mpl_toolkits.mplot3d import Axes3D
import time

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
if(jmax %2 == 0):
    jmax = jmax +1

print(SPACE_SIDE_LENGTH / ds)

# FIELDS

hx_field = np.zeros([jmax, jmax])
hy_field = np.zeros([jmax, jmax])
ez_field = np.zeros([jmax, jmax]) # multiply by sqrt(mu0 / eps0) to get the unit back
# TM Mode, so hz = 0

# SOURCE

jsource = int(jmax/2)
def source_function(t):
    f0 = 3e6
    w0 = 2 * np.pi * f0
    tau = 1/5
    return np.exp(-((t/TIME_SCALE - 1)**2/tau**2)) * np.sin(w0*t)


# PLOTS

fig = plt.figure()
ax = fig.add_subplot()
Explot = ax.pcolormesh(ez_field)
Explot.set_norm(matplotlib.colors.Normalize(vmin=-0.1, vmax=0.1))

# MATERIALS

eps = np.ones(jmax)
eps[int(3*jmax/4):] = 3

material_shape = eps > 1

# SIMULATION

simulation_steps_per_frame = int((frame_time * TIME_SCALE) / dt)
max_frames = int(SIMULATION_TIME / frame_time)
print((frame_time * TIME_SCALE) / dt)

current_n = 0
def update(_):
    global Explot, hy_field, current_n, ax
    for n in range(current_n, current_n+simulation_steps_per_frame):

        ez_field[1:jmax, 1:jmax] = ez_field[1:jmax, 1:jmax] + \
                                    0.5 * (hy_field[1:jmax, 1:jmax] - hy_field[:jmax-1, 1:jmax] - hx_field[1:jmax, 1:jmax] + hx_field[1:jmax, :jmax-1])
        
        ez_field[jsource, jsource] += source_function(n*dt)
        
        hx_field[:jmax-1, :jmax-1] = hx_field[:jmax-1, :jmax-1] + 0.5 * (ez_field[:jmax-1, :jmax-1] - ez_field[:jmax-1, 1:jmax])  
        hy_field[:jmax-1, :jmax-1] = hy_field[:jmax-1, :jmax-1] + 0.5 * (ez_field[1:jmax, :jmax-1] - ez_field[:jmax-1, :jmax-1])  

    current_n += simulation_steps_per_frame
    Explot.set_array(ez_field)
    ax.set_title(int(current_n * dt / TIME_SCALE))

animation = ani.FuncAnimation(fig=fig, func=update, frames=SIMULATION_TIME * FPS, interval=1000/FPS)
plt.show()