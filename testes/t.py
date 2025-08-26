import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import matplotlib

fig = plt.figure(figsize=[8,4])
fig.suptitle("E Field Magnitude: 0 ns")

ax1, ax2 = fig.subplots(1, 2, sharey=True)
Explot = ax1.pcolormesh([[0,0,0], [1,1,1]])
Explot.set_norm(matplotlib.colors.Normalize(vmin=0, vmax=0.01))
plt.show()