from scipy.constants import e, k, pi

config_dict = {

        # Geometry
        'R': 6e-2,
        #'R' : 1,
        #'L' : 0.5,
        'L': 10e-2,
        's': 1e-3,
        
        # Grid transparency   # TODO : are these still valid for N, do we need multiple values ?
        'beta_i': 0.7,
        'beta_g': 0.3,

        # potential difference
        'V_grid': 1000, 

        # Electrical
        'omega': 13.56e6 * 2 * pi,
        'N': 5,
        'R_coil': 2,
}
