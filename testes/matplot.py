import matplotlib.pyplot as plt
import numpy as np
import time

plt.ion()
x = np.linspace(0, 4 * np.pi, 500)
y = np.sin(x)
fig, ax = plt.subplots()
line, = ax.plot(x, y)

for i in range(500):
    if not plt.fignum_exists(fig.number):
        break
    
    y = np.sin(x - 0.1*i)
    line.set_ydata(y)
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.1)

plt.show()