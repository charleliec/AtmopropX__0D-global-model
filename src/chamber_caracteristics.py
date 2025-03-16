import numpy as np
from scipy.constants import pi, e, k, epsilon_0 as eps_0, c, m_e, N_A
from scipy.special import jv


# ! A check pour formules 
# ! + adapter code a la nouvelle version : variable désormais attributs de classe => self....

class Chamber(object):
    SIGMA_I = 1e-18 # Review this for iodine

    def __init__(self, config_dict):
        # Geometry
        self.R      = config_dict["R"]
        self.L      = config_dict["L"]
        self.s      = config_dict["s"]

        # Neutral flow
        self.m_i    = config_dict["m_i"]
        self.Q_g    = config_dict["Q_g"]
        self.beta_g = config_dict["beta_g"]
        self.kappa  = config_dict["kappa"]

        # Ions
        self.beta_i = config_dict["beta_i"]
        self.V_grid = config_dict["V_grid"]

        # Electrical
        self.omega  = config_dict["omega"]
        self.N      = config_dict["N"]
        self.R_coil = config_dict["R_coil"]
        self.I_coil = config_dict["I_coil"]

        # Initial values
        self.T_e_0  = config_dict["T_e_0"]
        self.n_e_0  = config_dict["n_e_0"]
        self.T_g_0  = config_dict["T_g_0"]


    def u_B(self,T_e, m_ion):
        """T_e in eV, m_ion is mass of single ion"""
        return np.sqrt(e*T_e/(N_A*m_ion))

    def h_L(self, n_g, L):
        lambda_i = 1/(n_g * self.SIGMA_I)
        return 0.86 / np.sqrt(3 + (L / (2 * lambda_i)))

    def h_R(self, n_g, R):
        lambda_i = 1/(n_g * self.SIGMA_I)
        return 0.8 / np.sqrt(4 + (R / lambda_i))

    def maxwellian_flux_speed(self, T, m):
        # TODO : what is it, where does it come from, does it always hold true
        return np.sqrt((8 * k * T) / (pi * m))

    def pressure(self, T, Q, v, A_out):
        """Calculates pressure in steady state without any plasma.
        T : Temperature in steady state
        Q : inflow rate in the chamber
        v : mean velocity of gas in steady state
        A_out : Effective area for which the gas can leave the chamber"""
        return (4 * k * T * Q) / (v * A_out)

    def A_eff(self, n_g, R, L): ### surface effective de perte des ions Xe+
        return (2 * self.h_R(n_g, R) * pi * R * L) + (2 * self.h_L(n_g, L) * pi * R**2)

    def A_eff_1(self, n_g, R, L, beta_i): ###surface effective sur laquelle les ions Xe+ sont recyclés en atomes Xe
        return 2 * self.h_R(n_g, R) * pi * R * L + (2 - beta_i) * self.h_L(n_g, L) * pi * R**2

    def eps_p(self, omega, n_e, n_g, K_el):
        omega_pe_sq = (n_e * e**2) / (m_e * eps_0)
        nu_m_i = 1j * K_el * n_g
        return 1 - (omega_pe_sq / (omega * (omega -  nu_m_i)))

    def gamma_ion(self, n_ion, m_ion, T_e, L):
        return self.h_L(n_ion, L) * n_ion * self.u_B(T_e, m_ion)

    def gamma_e(self, n_e, T_e):
        return n_e * self.u_B(T_e, m_e)

    def gamma_neutral(self, n_neutral, T_neutral, m_neutral):
        return n_neutral*np.sqrt(8*e*T_neutral/(pi*N_A*m_neutral))/4
