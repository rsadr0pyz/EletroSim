import numpy as np
from constants import *

def _initialize_dimension(ds, length, v0, axis):
            if v0 == None:
                size = int(length/ds) + 1
                length = int(length/ds) * ds
                shape = [1, 1, 1]
                shape[axis] = size
                return size, length, np.arange(size).reshape(shape)
            else:
                return 0, length, int(v0/ds)
            
# Contains all the relevant parameters for the calculation
class Parameters:
    
    def __init__(self, frequency, ds, m, n, width, height, depth=None,
                    x0=None, y0=None, z0=None):
        
        # If depth is not defined, then z position must be, and vice versa.
        assert (depth != None) ^ (z0 != None)

        self.ds = ds

        self.x_size, self.width, self.x = _initialize_dimension(ds, width, x0, 0)
        self.y_size, self.height, self.y = _initialize_dimension(ds, height, y0, 1)
        self.z_size, self.depth, self.z = _initialize_dimension(ds, depth, z0, 2)

        self.w0 = frequency*2*np.pi
        k = self.w0 * np.sqrt(EPS0*MU0)
        e_k_x = m * np.pi / width 
        e_k_y = n * np.pi / height

        self.e_gam = np.emath.sqrt(e_k_x ** 2 + e_k_y ** 2 - k ** 2)
        self.h2 = e_k_x ** 2 + e_k_y ** 2
        self.k = k
        self.e_k_x = e_k_x
        self.e_k_y = e_k_y

    
            
        