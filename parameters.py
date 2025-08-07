import numpy as np
from constants import *

# Contains all the relevant parameters for the calculation
class Parameters:
    
    def __init__(self, frequency, width, height, ds, m, n, depth=None):
        
        self.ds = ds

        self.x_size = int(width/ds) + 1 
        self.y_size = int(height/ds) + 1 

        self.width = int(width/ds) * ds
        self.height = int(height/ds) * ds

        if depth != None:
            self.z_size = int(depth/ds) + 1
            self.depth = int(depth/ds) * ds
        else:
            self.z_size = 1 # single plane simulation
            self.depth = 0
            
        print(f"Actual simulated dimensions: {self.width}, {self.height}, {self.depth}")
        print(f"matrix size: {self.x_size}, {self.y_size}, {self.z_size}")

        self.w0 = frequency*2*np.pi
        k = self.w0 * np.sqrt(EPS0*MU0)
        e_k_x = m * np.pi / width 
        e_k_y = n * np.pi / height

        self.e_gam = np.emath.sqrt(e_k_x ** 2 + e_k_y ** 2 - k ** 2)
        self.h2 = e_k_x ** 2 + e_k_y ** 2
        self.k = k
        self.e_k_x = e_k_x
        self.e_k_y = e_k_y

        print(f"gam: {self.e_gam}")