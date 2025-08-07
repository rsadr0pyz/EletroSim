import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from constants import *
from parameters import *
from fields import *

# PARAMETERS
m = 2
n = 1

width, height = 5, 5 # meters
frequency = 572.87e6 # hertz
ds = 1e-2 # meters

parameters = Parameters(frequency, ds, m, n, width=width, height=height, depth=5, y0=1.25)
e_field = EField(parameters)

print("SIMULATION SUMMARY")
print(f"k: {parameters.k}")
print(f"e_gam: {parameters.e_gam}")
print(f"e_k_x: {parameters.e_k_x}, e_k_y: {parameters.e_k_y}")
print(f"Actual simulated dimensions: {parameters.width}, {parameters.height}, {parameters.depth}")
print(f"matrix size: {parameters.x_size}, {parameters.y_size}, {parameters.z_size}")


sample = e_field.get_field_at(0)

sample = sample ** 2
sample = np.sum(sample, axis=3)
sample = np.sqrt(sample)
print(sample.shape)


fig = plt.figure()
axs = fig.subplots(3, sharex=True, sharey=True)

Explot = axs[0].pcolormesh(sample[:,0,:])
Explot.set_norm(matplotlib.colors.Normalize(vmin=0, vmax=5))

plt.show()