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

parameters = Parameters(frequency, width, height, ds, m, n)
e_field = EField(parameters)

print(parameters.e_k_x, parameters.e_k_y)
sample = e_field.get_field_at(0)
sample = sample ** 2
sample = np.sqrt(np.sum(sample, axis=2))

fig = plt.figure()
axs = fig.subplots(3, sharex=True, sharey=True)

Explot = axs[0].pcolormesh(sample[0])
Explot.set_norm(matplotlib.colors.Normalize(vmin=0, vmax=2))
Explot2 = axs[1].pcolormesh(sample[1])
Explot2.set_norm(matplotlib.colors.Normalize(vmin=0, vmax=2))
Explot3 = axs[2].pcolormesh(sample[2])
Explot3.set_norm(matplotlib.colors.Normalize(vmin=0, vmax=2))
plt.show()