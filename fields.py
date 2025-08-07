from parameters import *


class EField:

    def __init__(self, parameters: Parameters):
        self.parameters = parameters
        self.calculate_field_profile()

    def _field_at(self, y, x, z): # Line, column numpy pattern
        parameters = self.parameters
        e_k_x = parameters.e_k_x
        e_k_y = parameters.e_k_y
        e_gam = parameters.e_gam
        h2 = parameters.h2
        ds = parameters.ds

        ez = np.sin(x*ds*e_k_x) * np.sin(y*ds*e_k_y)
        ex = -e_gam / h2 * e_k_x * np.cos(x*ds*e_k_x) * np.sin(y*ds*e_k_y)
        ey = -e_gam / h2 * e_k_y * np.sin(x*ds*e_k_x) * np.cos(y*ds*e_k_y)
        field_profile = np.stack([ex, ey, ez], axis=-1)
        return field_profile * np.exp(-e_gam * z)
    
    def calculate_field_profile(self):
        parameters = self.parameters
        self.field_profile = np.fromfunction(self._field_at, [parameters.y_size, parameters.x_size, parameters.z_size])

    def get_field_at(self, z, t): 
        e_gam = self.parameters.e_gam
        w0 = self.parameters.w0
        z = np.array(z)[:,np.newaxis, np.newaxis, np.newaxis]
        field_phase_adjusted = self.field_profile * np.exp(-e_gam * z) * np.exp(w0*t * 1j)
        return np.real(field_phase_adjusted)