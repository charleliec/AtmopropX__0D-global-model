import numpy as np 
from numpy.typing import NDArray
from scipy.constants import m_e, e, pi, k, epsilon_0 as eps_0, mu_0   # k is k_B -> Boltzmann constant
from scipy.integrate import trapezoid, solve_ivp, odeint
from scipy.interpolate import interp1d

#Local modules
from src.util import load_csv, load_cross_section
from src.auxiliary_funcs import pressure, maxwellian_flux_speed, u_B, A_eff, A_eff_1, SIGMA_I, h_L
from src.specie import Specie, Species
from reactions.reaction import Reaction
from config import *

class GlobalModel:

    def __init__(self, species: Species, reaction_set: list[Reaction]):
        """Object simulating the evolution of a plasma with 0D model. 
            Inputs :
                config_dict : dictionary containing all parameters about the experimental setup
                species : instance of Species with all species being considered
                reaction_set : list with all reactions being considered -> [Reaction]"""
        self.load_chemistry()
        self.species = species
        self.reaction_set = reaction_set

    # def load_config(self, config_dict: dict[str, float]):

    #     # Geometry
    #     self.R = config_dict['R'] # Chamber's radius 
    #     self.L = config_dict['L'] # Length
        
    #     # Neutral flow
    #     self.m_i = config_dict['m_i']
    #     self.Q_g = config_dict['Q_g']
    #     self.beta_g = config_dict['beta_g']
    #     self.kappa = config_dict['kappa']

    #     # Ions
    #     self.beta_i = config_dict['beta_i'] # Transparency to ions (!!! no exact formula => to be determined later on...)
    #     self.V_grid = config_dict['V_grid'] # potential difference

    #     # Electrical
    #     self.omega = config_dict['omega']  # Pulsation of electro-mag fiel ?
    #     self.N = config_dict['N']  # Number of spires
    #     self.R_coil = config_dict['R_coil'] #Radius of coil around plasma
    #     self.I_coil = config_dict['I_coil'] 

    #     # Initial values = values in steady state before ignition of plasma
    #     self.T_e_0 = config_dict['T_e_0']
    #     self.n_e_0 = config_dict['n_e_0']
    #     self.T_g_0 = config_dict['T_g_0']
    # # gas density (in m^-3 ??) in steady state without plasma
    #       self.n_g_0 = pressure(self.T_g_0, self.Q_g,     
    #                               maxwellian_flux_speed(self.T_g_0, self.m_i),
    #                               self.A_g) / (k * self.T_g_0)
    
    

    def flux_i(self, T_e, T_g, n_e, n_g):
        """Ion flux leaving the thruster through the grid holes"""
        return h_L(n_g, self.L) * n_e * u_B(T_e, self.m_i)

    def thrust_i(self, T_e, T_g, n_e, n_g):
        """Total thrust produced by the ion beam"""
        return self.flux_i(T_e, T_g, n_e, n_g) * self.m_i * self.v_beam * self.A_i

    def j_i(self, T_e, T_g, n_e, n_g):
        """Ion current density extracted by the grids"""
        return self.flux_i(T_e, T_g, n_e, n_g) * e

    def eval_property(self, func, sol):
        """Calculates a property based on 'state' for all 't'.
            sol must be a np.array with shape (nb_of_t's, dimension_of_state) where each line represents a state"""
        prop = np.zeros(sol.shape[0])   #number of instants
        for i in np.arange(sol.shape[0]):
            prop[i] = func(sol[i])
        return prop
    
    ##def R_ind(R, L, N, omega, n_e, n_g, K_el):
    ##    ep = eps_p(omega, n_e, n_g, K_el)
    ##    k_p = (omega / c) * np.sqrt(ep)
    ##    a = 2 * pi * N**2 / (L * omega * eps_0)
    ##    b = 1j * k_p * R * jv(1, k_p * R) / (ep * jv(0, k_p * R))
    ##    return a * np.real(b)
    
    ##def P_abs(self, state):
        # ! n_g à changer
    ##    return R_ind(self.R, self.L, self.N, self.omega, n_e, n_g, self.K_el(T_e)) * self.I_coil**2 / 2 # the original code divided by V : density of power ?
    

    # def P_rf(self, state: NDArray[float]): # type: ignore
    #     """Calculates the power delivered by the coil using RF"""
    #     # ! To adapt to new var state
    #     T_e, T_g, n_e, n_g = state
    #     R_ind_val = R_ind(self.R, self.L, self.N, self.omega, n_e, n_g, self.K_el(T_e))
    #     return (1/2) * (R_ind_val + self.R_coil) * self.I_coil**2
    def normalised_concentrations(self , state , species) :
        """Établissement d'une liste normalisée des concentrations des espèces (telle que leur somme vaut 1)"""
        total_c , N = 0 , species.nb
        normalised_c = []
        for i in range(1 , N):
            total_c += state[i]
        for i in range(1 , N):
            normalised_c.append(state[i]/total)
        return normalised_c
        
    def eps_p (liste_eps , liste_c) :
        """calcule la permittivité diélectrique relative due à toutes les réactions de collisions ellastiques elctron-neutre. Ces réactions sont considérées séparément par chacun des eps_i"""
        def equation(x):
            sum = 0
            for i in(range(len(liste_eps))):
                sum += liste_[i](liste_eps[i]-1)/(liste_eps[i] + 2*x)
            return sum + (1-x)/(3*x)
        return fsolve(equation , 1)

    
    def f_dy(self, t, state):
        """Returns the derivative of the vector 'state' describing the state of plasma.
            'state' has format : [n_e, n_N2, ..., n_N+, T_e, T_monoato, ..., T_diato]"""
        densities = state[:self.species.nb]
        temp_by_sp = [state[self.species.nb + sp.nb_atoms] for sp in self.species.list]

        dy = np.zeros(state.shape)
        dy_densities = np.zeros(self.species.nb)
        dy_energies = np.zeros(3)

        eps_i_list = []
        normalised_c = normalised_concentrations(state)

        for reac in self.reaction_set:
            dy_densities += reac.density_change_rate(state)
            dy_energies += reac.energy_change_rate(state)
            if reac.is_elastic_collision :
                eps_i_list.append(reac.get_eps_i(state))
        eps_p = eps_p(eps_i_list , normalised_c)

        # calculation of R_ind with intermediary steps
        k_p = (omega / c) * np.sqrt(eps_p)
        a = 2 * pi * N**2 / (L * omega * eps_0)
        #jv are Besel functions
        b = 1j * k_p * R * jv(1, k_p * R) / (eps_p * jv(0, k_p * R))
        R_ind = a * np.real(b)
        #deducing P_abs from R_ind
        P_abs = R_ind* self.I_coil**2 / 2

        # Energy given to the electrons via the coil
        dy_energies[0] += self.P_abs(state)
        #Transform derivative of energy into derivative of temperature
        dy_temp = 2/3 * dy_energies/(e*densities) - dy_densities*temp_by_sp/densities

        dy[:self.species.nb] = dy_densities
        dy[self.species.nb:] = dy_temp

        return dy
    

    def solve(self, t0, tf):
        y0 = np.array([self.T_e_0, self.T_g_0, self.n_e_0, self.n_g_0])
        return solve_ivp(self.f_dy, (t0, tf), y0, method='LSODA')


    def solve_for_I_coil(self, I_coil):
        """Calculates for a list of intensity in the coil the resulting power consumption and the resulting thrust.
            ## Returns
            power_array , list_of(`state` after long time)"""
        p = np.zeros(I_coil.shape[0])
        solution = np.zeros((I_coil.shape[0], 4))  #shape = (y,x)

        for i, I in enumerate(I_coil):
            self.I_coil = I

            sol = self.solve(0, 5e-2)    # TODO Needs some testing

            final_state = sol.y[:, -1]

            p[i] = self.P_rf(final_state)

            solution[i] = final_state
            
        return p, solution
