from scipy.constants import e, k, pi
import numpy as np


# ! Check la pertinence des valeurs (issus du code initial ptet que valable pour Xe...)

config_dict = {

        # Geometry
        'R': 1e-2,
        'L': 18e-2,
        
        # Neutral flow
        'pressure': 30.66, #in Pa
        'kappa': 0.0057,

        # Electrical
        'omega': 40.66e6 * 2 * pi,
        'V_grid': 1000,  # potential difference
        'N': 1.5,
        'R_coil': 1.21e-4

}


# # ! A vérifier
# # Geometry
# R = 6e-2
# L = 10e-2
# s = 1e-3

# # Neutral flow
# m_i = 2.18e-25
# Q_g = 1.2e19
# beta_g = 0.3
# kappa = 0.0057

# # Ions
# beta_i = 0.7
# V_grid = 1000  # potential difference

# # Electrical
# omega = 13.56e6 * 2 * pi
# N = 5
# R_coil = 2
# I_coil = 26

# # Initial values
# T_e_0 = 3 * e / k
# n_e_0 = 1e18
# T_g_0 = 300


# # @property
# # def A_g(self): return self.beta_g * pi * self.R**2

# # @property
# # def A_i(self): return self.beta_i * pi * self.R**2


# V_chamber = pi * R**2 * L 

# @property
# def A(self): return 2*pi*self.R**2 + 2*pi*self.R*self.L

# @property
# def v_beam(self): 
#     """Ion beam's exit speed"""
#     return np.sqrt(2 * e * self.V_grid / self.m_i)



# config_dict = {

#         # Geometry
#         'R': 6e-2,
#         'L': 10e-2,
#         's': 1e-3,
        
#         # Neutral flow
#         'm_i': 2.18e-25,
#         'Q_g': 1.2e19,
#         'beta_g': 0.3,
#         'kappa': 0.0057,

#         # Ions
#         # TODO : are these still valid for N, do we need multiple values ?
#         'beta_i': 0.7,
#         'V_grid': 1000,  # potential difference

#         # Electrical
#         'omega': 13.56e6 * 2 * pi,
#         'N': 5,
#         'R_coil': 2,
#         'I_coil': 26,

#         # Initial values
#         'T_e_0': 3 * e / k,
#         'n_e_0': 1e18,
#         'T_g_0': 300

# }

