import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib

class DiscreteVectorField:
    
    def __init__(self, side_length: int):
        self.side_length = side_length

        # X, Y, Z (side_length) positions + 3 components (x,y,z) for each vector
        self.Fx = np.zeros([side_length, side_length, side_length]) #np.random.randint(low=-5, high=5, size=[side_length]*3)
        self.Fy = np.zeros([side_length, side_length, side_length]) #np.random.randint(low=-5, high=5, size=[side_length]*3)
        self.Fz = np.zeros([side_length, side_length, side_length]) #np.random.randint(low=-5, high=5, size=[side_length]*3)
        self.curl = np.zeros([side_length-1, side_length-1, side_length-1, 3])

    def compute_curl(self):
        self.pypx = np.gradient(self.Fy, axis=0)
        self.pzpx = np.gradient(self.Fz, axis=0)

        self.pxpy = np.gradient(self.Fx, axis=1)
        self.pzpy = np.gradient(self.Fz, axis=1)

        self.pxpz = np.gradient(self.Fx, axis=2)
        self.pypz = np.gradient(self.Fy, axis=2)
        
        x_comp = self.pzpy - self.pypz
        y_comp = self.pxpz - self.pzpx
        z_comp = self.pypx - self.pxpy
        
        np.stack((x_comp, y_comp, z_comp), axis=-1, out=self.curl)

    def curl_at(self, pos):
        x_comp = self.pzpy[*pos] - self.pypz[*pos]
        y_comp = self.pxpz[*pos] - self.pzpx[*pos]
        z_comp = self.pypx[*pos] - self.pxpy[*pos]

        return np.array([x_comp, y_comp, z_comp])


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

h_field = DiscreteVectorField(jmax)
e_field = DiscreteVectorField(jmax-1)

# PLOTS

fig = plt.figure()
ax = fig.add_subplot()
Explot = ax.pcolormesh(ez)
Explot.set_norm(matplotlib.colors.Normalize(vmin=-0.1, vmax=0.1))


# SOURCE

jsource = int(jmax/2)
def source_function(t):
    f0 = 3e6
    w0 = 2 * np.pi * f0
    tau = 1/5
    return np.exp(-((t/TIME_SCALE - 1)**2/tau**2)) * np.sin(w0*t)

# SIMULATION

simulation_steps_per_frame = int((frame_time * TIME_SCALE) / dt)
max_frames = int(SIMULATION_TIME / frame_time)
print((frame_time * TIME_SCALE) / dt)

current_n = 0
def update(_):
    global h_field, ez, current_n
    for n in range(current_n, current_n+simulation_steps_per_frame):
        h_field.compute_curl()
        
        ez[0:jmax-1, 0:jmax-1] = ez[0:jmax-1, 0:jmax-1] - 0.5 * h_field.curl
        ez[jsource, jsource] += source_function(n*dt)

    current_n += simulation_steps_per_frame
    Explot.set_array(ez_field)
    ax.set_title(int(current_n * dt / TIME_SCALE))

animation = ani.FuncAnimation(fig=fig, func=update, frames=SIMULATION_TIME * FPS, interval=1000/FPS)
plt.show()