import numpy as np
from scipy.constants import pi, e, k, epsilon_0 as eps_0, c, m_e, N_A
from scipy.special import jv
from src.specie import Specie, Species

# ! A check pour formules 
# ! + adapter code a la nouvelle version : variable désormais attributs de classe => self....

class Chamber(object):
    SIGMA_I = 1e-18 # Review this for iodine

    def __init__(self, config_dict):
        # Geometry
        self.R      = config_dict["R"]
        self.L      = config_dict["L"]
        self.s      = config_dict["s"]
        self.S_grid = pi * self.R**2
        self.S_walls = pi * self.R * self.L
        self.S_total = 2 * self.S_grid + self.S_walls

        # Neutral flow
        self.Q_g    = config_dict["Q_g"]
        self.beta_g = config_dict["beta_g"]
        self.kappa  = config_dict["kappa"]

        # Ions
        self.beta_i = config_dict["beta_i"]

        # Electrical
        self.omega  = config_dict["omega"]
        self.N      = config_dict["N"]
        self.R_coil = config_dict["R_coil"]
        self.I_coil = config_dict["I_coil"]
        self.V_grid = config_dict["V_grid"]


        # Initial values
        self.T_e_0  = config_dict["T_e_0"]
        self.n_e_0  = config_dict["n_e_0"]
        self.T_g_0  = config_dict["T_g_0"]

    
    
    
    def u_B(self,T_e, m_ion):
        """T_e in eV, m_ion is mass of single ion"""
        return np.sqrt(e*T_e/(m_ion))

    def h_L(self, n_g):
        lambda_i = 1/(n_g * self.SIGMA_I)
        return 0.86 / np.sqrt(3 + (self.L / (2 * lambda_i)))

    def h_R(self, n_g):
        lambda_i = 1/(n_g * self.SIGMA_I)
        return 0.8 / np.sqrt(4 + (self.R / lambda_i))

    def maxwellian_flux_speed(self, T, m):
        return np.sqrt((8 * e * T) / (pi * m))

    def v_beam(self , m_ion , charge) :
        '''beam speed on an ion going through the grids'''
        return np.sqrt((2*charge*self.V_grid)/m_ion)

    def pressure(self, T, v, A_out):
        """Calculates pressure in steady state without any plasma.
        T : Temperature in steady state
        v : mean velocity of gas in steady state"""
        return (4 * k * T * self.Q_g) / (v * self.beta_i * self.S_grid)
    
    def S_eff_neutrals(self):
        """Effective surface through which neutrals leaves the chanber through the grids"""
        return self.beta_g * self.S_grid
    
    # TODO : n_g c'est quoi ? densité du gaz considéré ou tout le gaz ?

    def S_eff_total(self, n_g):
        """Total effective surface on which ions and electrons are lost are lost. Equals hS.
         For ions takes into account ions lost through the grids and ions neutralized at the walls """
        return (2 * self.h_R(n_g) * pi * self.R * self.L) + (2 * self.h_L(n_g) * pi * self.R**2)

    def S_eff_total_ion_neutrelisation(self, n_g):
        """Effective area on which ions are neutralized. Equals S_eff_total - beta_i * S_grid."""
        return 2 * self.h_R(n_g) * pi * self.R * self.L + (2 - self.beta_i) * self.h_L(n_g) * pi * self.R**2


    def gamma_ion(self, n_ion, T_e , m_ion):
        return self.h_L(n_ion) * n_ion * self.u_B(T_e, m_ion)


    def gamma_neutral(self, n_neutral, T_neutral, m_neutral):
        return n_neutral*np.sqrt(8*e*T_neutral/(pi*N_A*m_neutral))/4
