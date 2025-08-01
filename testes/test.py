import numpy as np
import matplotlib.pyplot as plt
import math
import time

eps0 = 8.8541878128e-12
mu0 = 1.256637062 * 1e-6
c = 1 / math.sqrt(mu0 * eps0)
imp0 = math.sqrt(mu0 / eps0)



jmax = 500
nmax = 10000

Ex = np.zeros(jmax)
Hz = np.zeros(jmax)
Ex_prev = np.zeros(jmax)
Hz_prev = np.zeros(jmax)

lambda_min = 350e-9 #meters
dx = lambda_min/20
dt = dx / c

eps = np.ones(jmax) * eps0
eps[250:300] = 10 * eps0

material_shape = eps > eps0

jsource = 100
def source_function(t):
    lambda_0 = 550e-9
    w0 = 2 * np.pi * c / lambda_0
    tau = 30
    t0 = 3 * tau
    return np.exp(-((t-t0)**2/tau**2)) * np.sin(w0*t*dt)

plt.ion()
fig, ax = plt.subplots()
Explot, = ax.plot(Ex)
ax2 = ax.twinx()
Hzplot, = ax2.plot(Hz)
Hzplot.set_color('tab:red')
ax.plot(material_shape)
ax.set_ylim([-1, 1])
ax2.set_ylim([-0.01, 0.01])

for n in range(nmax):
    if not plt.fignum_exists(fig.number):
        break

    #update magnetic field

    #boundaries
    Hz[jmax-1] = Hz_prev[jmax - 2]
    Hz[:jmax-1] = Hz_prev[:jmax-1] + dt / (dx * mu0) * (Ex[1:jmax] - Ex[:jmax-1])
    Hz_prev = Hz
        

    Hz[jsource-1] -= source_function(n)/imp0
    Hz_prev[jsource-1] = Hz[jsource-1]

    #update eletric field

    # boundaries
    Ex[0] = Ex_prev[1]
    Ex[1:] = Ex_prev[1:] + dt / (dx * eps[1:]) * (Hz[1:] - Hz[:jmax-1])
    Ex_prev = Ex

    #sources
    Ex[jsource] += source_function(n+1)
    Ex_prev[jsource] = Ex[jsource]

    #plot
    if n % 10 == 0:
        Explot.set_ydata(Ex)
        Hzplot.set_ydata(Hz)
        fig.canvas.draw()
        fig.canvas.flush_events()
