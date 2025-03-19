import numpy as np 
from numpy.typing import NDArray
from scipy.constants import m_e, e, pi, k, epsilon_0 as eps_0, mu_0   # k is k_B -> Boltzmann constant
from scipy.integrate import trapezoid, solve_ivp, odeint
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

#Local modules
from src.util import load_csv, load_cross_section
# from src.auxiliary_funcs import pressure, maxwellian_flux_speed, u_B, A_eff, A_eff_1, SIGMA_I, h_L
from src.specie import Specie, Species
from src.reactions.reaction import Reaction
from src.reactions.general_elastic_collision import GeneralElasticCollision
from src.chamber_caracteristics import Chamber

class GlobalModel:

    def __init__(self, species: Species, reaction_set: list[Reaction], chamber: Chamber):
        """Object simulating the evolution of a plasma with 0D model. 
            Inputs :
                config_dict : dictionary containing all parameters about the experimental setup
                species : instance of Species with all species being considered
                reaction_set : list with all reactions being considered -> [Reaction]"""
        self.species = species
        self.reaction_set = reaction_set

    def flux_i(self, T_e, T_g, n_i, n_g): # TODO vérifier j'ai remplacé n_e par n_i car il faut prendre en compte la quantité d'ions et pas d'électrons
        """Ion flux leaving the thruster through the grid holes"""
        return h_L(n_g, self.L) * n_i * u_B(T_e, self.m_i)

#TODO vérifier on devrait aussi prendre en compte le flux sortant de particules neutres pour suivre le modèle de Chabert
    def flux_n(self, T_e, T_g, n_i, n_g):
    """Neutral flux leaving the thruster through grid holes"""
        return 1/4 * n_g * maxwellian_flux_speed(T_g, self.m_i)
    
    def thrust_i(self, T_e, T_g, n_i, n_g):
        """Total thrust produced by the ion beam"""
        return self.flux_i(T_e, T_g, n_i, n_g) * self.m_i * self.v_beam * self.S_ion

# TODO vérifier je rajoute le thrust produit par les particules neutres
    def thrust_n(self, T_e, T_g, n_i, n_g):
        """Total thrust produced by neutral atoms"""
        return self.flux_n(T_e, T_g, n_i, n_g) * self.m_i * self.v_beam *self.S_neutre

    def j_i(self, T_e, T_g, n_i, n_g):
        """Ion current density extracted by the grids"""
        return self.flux_i(T_e, T_g, n_i, n_g) * e

    def eval_property(self, func, sol):
        """Calculates a property based on 'state' for all 't'.
            sol must be a np.array with shape (nb_of_t's, dimension_of_state) where each line represents a state"""
        prop = np.zeros(sol.shape[0])   #number of instants
        for i in np.arange(sol.shape[0]):
            prop[i] = func(sol[i])
        return prop
    
    def normalised_concentrations(self , state: NDArray[np.float64]) :
        """Returns normalized concentrations (such that they sum up to 1)"""
        return state[:self.species.nb]/np.sum(state[:self.species.nb])
        
# TODO vérifier parce que la formule est fausse (eps_p est un complexe et pas un réel) et parce que je crois qu'on a déjà codé ça dans CollisionWithElectron...
    def eps_p (self, collision_frequencies , state) :
        """Calcule la permittivité diélectrique relative due à toutes les réactions de collisions elastiques elctron-neutre. Ces réactions sont considérées séparément par chacun des eps_i
            collision_frequencies : np.array containing the collision frequencies for each specie in the order in which they appear in self.species"""
        normalized_c = self.normalised_concentrations(state)
        omega_pe_sq = (state[0] * e**2) / (m_e * eps_0)
        epsilons_i = 1 - omega_pe_sq / (self.chamber.omega * (self.chamber.omega -  collision_frequencies)) ### TODO on soustrait un np.array à un float ??

        def equation(x):            
            return np.sum(normalized_c*(epsilons_i-1)/(epsilons_i + 2*x)) + (1-x)/(3*x) # TODO elle sort d'où dans Chabert cette expression ??
        return fsolve(equation , 1)

    
    def f_dy(self, t, state):
        """Returns the derivative of the vector 'state' describing the state of plasma.
            'state' has format : [n_e, n_N2, ..., n_N+, T_e, T_monoato, ..., T_diato]"""
        densities = state[:self.species.nb]
        temp = state[self.species.nb:]

        dy = np.zeros(state.shape)
        dy_densities = np.zeros(self.species.nb)
        dy_energies = np.zeros(3)

        collision_frequencies = np.zeros(self.species.nb)
        for reac in self.reaction_set:
            dy_densities += reac.density_change_rate(state)
            dy_energies += reac.energy_change_rate(state)
            if isinstance(reac, GeneralElasticCollision) :
                sp_idx, freq = reac.colliding_specie_and_collision_frequency(state)
                collision_frequencies[sp_idx] = freq
        eps_p = self.eps_p(collision_frequencies, state)

        # calculation of R_ind with intermediary steps
        k_p = (omega / c) * np.sqrt(eps_p)
        a = 2 * pi * N**2 / (L * omega * eps_0)
        #jv are Besel functions
        b = 1j * k_p * R * jv(1, k_p * R) / (eps_p * jv(0, k_p * R))
        R_ind = a * np.real(b)
        #deducing P_abs from R_ind
        P_abs = R_ind* self.I_coil**2 / 2

        # Energy given to the electrons via the coil
        dy_energies[0] += self.P_abs(state) # TODO où est P_loss ??

        # total thermal capacity of all species with same number of atoms : sum of (3/2 or 5/2 * density)
        total_thermal_capacity_by_sp_type = np.zeros(3)
        dy_total_thermal_capacity_by_sp_type = np.zeros(3)
        for sp in self.species.species :
            total_thermal_capacity_by_sp_type[sp.nb_atoms] += sp.thermal_capacity * densities[sp.index]
            dy_total_thermal_capacity_by_sp_type[sp.nb_atoms] += sp.thermal_capacity * dy_densities[sp.index]
            
        #Transform derivative of energy into derivative of temperature
        dy_temp = (dy_energies - temp * dy_total_thermal_capacity_by_sp_type) / total_thermal_capacity_by_sp_type

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
