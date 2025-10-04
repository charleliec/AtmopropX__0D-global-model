import numpy as np
from scipy.constants import pi, e, k, epsilon_0 as eps_0, c, m_e, N_A
from scipy.special import jv
from global_model_package.specie import Specie, Species

# ! A check pour formules
# ! + adapter code a la nouvelle version : variable désormais attributs de classe => self....

class Chamber(object):
    SIGMA_I = 1e-18 # Review this for iodine

    def __init__(self, config_dict):
        # Geometry
        self.R: float = config_dict["R"]
        self.L: float = config_dict["L"]
        self.V_chamber = pi * self.R**2 * self.L
        #self.V_chamber=0.5
        #self.s      = config_dict["s"]
        self.S_gridded_wall = pi * self.R**2
        self.S_cylindrical_wall: float = 2 * pi * self.R * self.L
        self.S_total = 2 * self.S_gridded_wall + self.S_cylindrical_wall

        # Electrical heating
        self.omega  = config_dict["omega"]
        self.N      = config_dict["N"]
        self.R_coil = config_dict["R_coil"]
        if "I_coil" in config_dict:
            self.I_coil = config_dict["I_coil"]
            print("Chamber init : 'I_coil' has been set, heating with constant current")

        # Thruster properties  / Kim chamber properties
        if all(k in config_dict for k in ("V_grid", "beta_g", "beta_i")):
            self.V_grid = config_dict["V_grid"]
            self.beta_g = config_dict["beta_g"]   # Neutral flow
            self.beta_i = config_dict["beta_i"]   # Ions
            print("Chamber init : 'V_grid', 'beta_g', 'beta_i' have been set, gas leaving chamber through grids")
        elif "target_pressure" in config_dict:
            self.target_pressure = config_dict["target_pressure"]
            print("Chamber init : 'target_pressure' has been set, gas ejected such that constant pressure is maintained")
        else:
            raise KeyError("No gas ejection method has been made possible in the config file. Please configure ('V_grid', 'beta_g', 'beta_i') or 'target_pressure'.")


    def u_B(self,T_e, m_ion):
        """T_e in eV, m_ion is mass of single ion"""
        return np.sqrt(e*T_e/(m_ion))

    def h_L(self, n_g_tot):
        lambda_i = 1/(n_g_tot * self.SIGMA_I)
        return 0.86 / np.sqrt(3 + (self.L / (2 * lambda_i)))

    def h_R(self, n_g_tot):
        lambda_i = 1/(n_g_tot * self.SIGMA_I)
        return 0.8 / np.sqrt(4 + (self.R / lambda_i))

    def maxwellian_flux_speed(self, T, m):
        return np.sqrt((8 * e * T) / (pi * m))

    def v_beam(self , m_ion , charge) :
        '''beam speed on an ion going through the grids'''
        return np.sqrt((2*charge*self.V_grid)/m_ion)


    def S_eff_neutrals(self):
        """Effective surface through which neutrals leaves the chanber through the grids"""
        return self.beta_g * self.S_gridded_wall

    def S_eff_total(self, n_g_tot):
        """Total effective surface on which ions and electrons are lost are lost. Equals hS.
         For ions takes into account ions lost through the grids and ions neutralized at the walls """
        return (2 * self.h_R(n_g_tot) * pi * self.R * self.L) + (2 * self.h_L(n_g_tot) * pi * self.R**2)

    def S_eff_total_ion_neutrelisation(self, n_g_tot):
        """Effective area on which ions are neutralized. Equals S_eff_total - beta_i * S_gridded_walls."""
        return 2 * self.h_R(n_g_tot) * pi * self.R * self.L + (2 - self.beta_i) * self.h_L(n_g_tot) * pi * self.R**2     


    def gamma_ion(self, n_ion, T_e , m_ion):
        return n_ion * self.u_B(T_e, m_ion)

    def gamma_neutral(self, n_neutral, T_neutral, m_neutral):
        return n_neutral*np.sqrt(8*e*T_neutral/(pi*m_neutral))/4