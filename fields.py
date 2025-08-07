from parameters import *


class EField:

    def __init__(self, parameters: Parameters):
        self.parameters = parameters
        self.calculate_field_profile()
    
    def calculate_field_profile(self):
        parameters = self.parameters
        e_k_x = parameters.e_k_x
        e_k_y = parameters.e_k_y
        e_gam = parameters.e_gam
        h2 = parameters.h2
        ds = parameters.ds
        x = parameters.x
        y = parameters.y
        z = parameters.z

        ez = np.sin(x*ds*e_k_x) * np.sin(y*ds*e_k_y) * np.exp(-z*ds*e_gam) 
        ex = -e_gam / h2 * e_k_x * np.cos(x*ds*e_k_x) * np.sin(y*ds*e_k_y) * np.exp(-z*ds*e_gam)
        ey = -e_gam / h2 * e_k_y * np.sin(x*ds*e_k_x) * np.cos(y*ds*e_k_y) * np.exp(-z*ds*e_gam)
        self.field = np.stack([ex, ey, ez], axis=-1)
    
    def get_field_at(self, t): 
        w0 = self.parameters.w0
        field_phase_adjusted = self.field * np.exp(w0*t * 1j)
        return np.real(field_phase_adjusted)