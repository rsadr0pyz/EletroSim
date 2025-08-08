import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from constants import *
from parameters import *
from fields import *

# PARAMETERS
m = 1
n = 1

width, height = 5, 1 # meters
frequency = 160e6 # hertz
frequency2 = 260e6 # hertz
ds = [1e-2, 1e-2, 1e-2] # meters
dt = 1

parameters = Parameters(frequency, ds, m, n, width=width, height=height, z0=0)
new_z=np.pi/2/np.abs(parameters.e_gam)
parameters1 = Parameters(frequency, ds, m, n, width=width, height=height, z0=new_z)
e_field = EFieldGuide(parameters)
e_field1 = EFieldGuide(parameters1)

parameters2 = Parameters(frequency2, ds, m, n, width=width, height=height, z0=0)
new_z=np.pi/2/np.abs(parameters2.e_gam)
parameters3 = Parameters(frequency2, ds, m, n, width=width, height=height, z0=new_z)
e_field2 = EFieldGuide(parameters2)
e_field3 = EFieldGuide(parameters3)

print("SIMULATION SUMMARY")
print(f"k: {parameters.k}")
print(f"e_gam: {parameters.e_gam}")
print(f"e_k_x: {parameters.e_k_x}, e_k_y: {parameters.e_k_y}")
print(f"Actual simulated dimensions: {parameters.width}, {parameters.height}, {parameters.depth}")
print(f"matrix size: {parameters.x_size}, {parameters.y_size}, {parameters.z_size}")


sample = e_field.get_field_at(0)
sample = sample ** 2
sample = np.sum(sample, axis=3)
sample = np.sqrt(sample)[0,:,:]

sample1 = e_field1.get_field_at(0)
sample1 = sample1 ** 2
sample1 = np.sum(sample1, axis=3)
sample1 = np.sqrt(sample1)[0,:,:]

sample2 = e_field2.get_field_at(0)
sample2 = sample2 ** 2
sample2 = np.sum(sample2, axis=3)
sample2 = np.sqrt(sample2)[0,:,:]

sample3 = e_field3.get_field_at(0)
sample3 = sample3 ** 2
sample3 = np.sum(sample3, axis=3)
sample3 = np.sqrt(sample3)[0,:,:]

max = np.max(sample)


fig = plt.figure()
axs = fig.subplots(2, 2, sharex=True, sharey=True)#fig.subplots(3, sharex=True, sharey=True)

Explot = axs[0][0].pcolormesh(np.arange(parameters.x_size)*ds[0], np.arange(parameters.y_size)*ds[1], sample)
Explot1 = axs[0][1].pcolormesh(np.arange(parameters.x_size)*ds[0], np.arange(parameters.y_size)*ds[1], sample1)
Explot2 = axs[1][0].pcolormesh(np.arange(parameters.x_size)*ds[0], np.arange(parameters.y_size)*ds[1], sample2)
Explot3 = axs[1][1].pcolormesh(np.arange(parameters.x_size)*ds[0], np.arange(parameters.y_size)*ds[1], sample3)
axs[0][0].set_title("Magnitude do Campo Elétrico (z=0) (160 Ghz)")
axs[1][0].set_xlabel("X (m)")
axs[0][0].set_ylabel("Y (m)")
axs[1][1].set_xlabel("X (m)")
axs[1][0].set_ylabel("Y (m)")
axs[0][1].set_title(f"Magnitude do Campo Elétrico (z=λ/2) (160 Ghz)")
axs[1][0].set_title("Magnitude do Campo Elétrico (z=0) (260 Ghz)")
axs[1][1].set_title(f"Magnitude do Campo Elétrico (z=λ/2) (260 Ghz)")

fig.tight_layout(pad=0.4)

Explot.set_norm(matplotlib.colors.Normalize(vmin=0, vmax=max))
Explot1.set_norm(matplotlib.colors.Normalize(vmin=0, vmax=max))

Explot2.set_norm(matplotlib.colors.Normalize(vmin=0, vmax=max))
Explot3.set_norm(matplotlib.colors.Normalize(vmin=0, vmax=max))
plt.show()