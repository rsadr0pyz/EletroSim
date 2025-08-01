import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani

t1 = np.random.rand(3,3)
FPS = 1

fig, ax = plt.subplots()

plot = ax.pcolormesh(t1)
print(t1)

def update(_):
    t1 = np.random.rand(3,3)
    plot.set_array(t1)

animation = ani.FuncAnimation(fig=fig, func=update, frames=FPS * 10, interval=1/FPS*1000)
plt.show()