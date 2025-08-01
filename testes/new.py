import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as ani
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
ds = c0 / (MAXIMUM_FREQUENCE * 40)
dt = ds/(2 * c0)
jmax = int(SPACE_SIDE_LENGTH  / ds)
print(SPACE_SIDE_LENGTH / ds)

# FIELDS

ex_field = np.zeros([jmax]) # multiply by sqrt(mu0 / eps0) to get the unit back
hy_field = np.zeros([jmax])
ex_1 = np.zeros([2])
ex_jmax = np.zeros([2])

# SOURCES

jsource = int(jmax/2)
def source_function(t):
    f0 = 3e6
    w0 = 2 * np.pi * f0
    tau = 1/5
    return np.exp(-((t/TIME_SCALE - 1)**2/tau**2)) * np.sin(w0*t)


# PLOTS

fig, ax = plt.subplots()
Explot, = ax.plot(ex_field)

ax.set_ylim(-2, 2)

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
    global hy_field, ex_field, current_n, ex_1
    for n in range(current_n, current_n+simulation_steps_per_frame):

        ex_field[1:jmax] = ex_field[1:jmax] + 0.5 / eps[1:jmax] * (hy_field[:jmax-1] - hy_field[1:jmax])
        ex_field[jsource] += source_function(n*dt)

        #Fronteiras
        ex_field[0] = ex_1[0]
        ex_1[0] = ex_1[1]
        ex_1[1] = ex_field[1]

        ex_field[jmax-1] = ex_jmax[0]
        ex_jmax[0] = ex_jmax[1]
        ex_jmax[1] = ex_field[jmax-2]

        hy_field[jsource-1] += source_function((n-1/2)*dt)
        hy_field[:jmax-1] = hy_field[:jmax-1] + 0.5 / eps[1:jmax] * (ex_field[:jmax-1] - ex_field[1:jmax])  


    current_n += simulation_steps_per_frame
    Explot.set_ydata(ex_field)
    ax.set_title(int(current_n * dt / TIME_SCALE))

animation = ani.FuncAnimation(fig=fig, func=update, frames=SIMULATION_TIME * FPS, interval=1000/FPS)
plt.show()