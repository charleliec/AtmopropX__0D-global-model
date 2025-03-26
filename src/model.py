import numpy as np 
from numpy.typing import NDArray
from scipy.constants import m_e, e, pi, k, epsilon_0 as eps_0, mu_0, c as c_light   # k is k_B -> Boltzmann constant
from scipy.integrate import trapezoid, solve_ivp, odeint
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy.special import jv

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
        self.chamber = chamber
    


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
        
    def eps_p (self, collision_frequencies , state) :
        """Calcule la permittivité diélectrique relative due à toutes les réactions de collisions elastiques elctron-neutre. Ces réactions sont considérées séparément par chacun des eps_i
            collision_frequencies : np.array containing the collision frequencies for each specie in the order in which they appear in self.species"""
        #fonction utilisée dans f_dy
        normalized_c = self.normalised_concentrations(state)
        omega_pe_sq = (state[0] * e**2) / (m_e * eps_0)
        epsilons_i = 1 - omega_pe_sq / (self.chamber.omega * (self.chamber.omega -  1j*collision_frequencies))

        def equation(x):            
            return np.sum(normalized_c*(epsilons_i-1)/(epsilons_i + 2*x)) + (1-x)/(3*x)
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
                collision_frequencies[sp_idx] += freq
        eps_p = self.eps_p(collision_frequencies, state)

        # calculation of P_abs : the power given by the antenna to the plasma
        power = self.P_abs(self.R_ind( eps_p  ))

        # Energy given to the electrons via the coil
        dy_energies[0] += power

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
    
    def R_ind(self, eps_p):
        '''plamsma resistance, used in calculating the power P_abs'''
        k_p = (self.chamber.omega / c_light) * np.sqrt(eps_p)
        a = 2 * pi * self.chamber.N**2 / (self.L * self.chamber.omega * eps_0)
        #jv are Besel functions
        b = 1j * k_p * self.chamber.R * jv(1, k_p * self.chamber.R) / (eps_p * jv(0, k_p * self.chamber.R))
        return a * np.real(b)

    def P_abs(self , R_ind):
        return R_ind* self.chamber.I_coil**2 / 2

    def thrust_i(self, T_e, n_e, n_ion , m_ion , charge):
        """Thrust produced by the ion beam of one specie"""
        return self.chamber.gamma_ion(n_ion, T_e , m_ion) * specie.mass * self.chamber.v_beam(m_ion , charge) * self.chamber.beta_i * pi * self.chamber.R ** 2

    def j_i(self, T_e, n_e, n_ion , m_ion , charge):
        """Ion current density of one ionic specie extracted by the grids"""
        return self.chamber.gamma_ion( n_ion, T_e, m_ion) * e * charge

        
    def total_ion_thrust(self , state ) :
        '''Calculates the total amount of thrust generated'''
        total_thrust = 0
        for i in(range(1,len(state)/2)):
            if self.species.species[i].charge != 0 :
                total_thrust += self.thrust_i( state[len(state)/2] , state[0] , state[i] , slef.species.species[i].mass , self.species.species[i].charge)


    def total_ion_current(self , state ) :
        '''Calculates the total amount of ion current toxards the grids'''
        total_current = 0
        for i in(range(1,len(state)/2)):
            if self.species.species[i].charge != 0 :
                total_current += self.j_i( state[len(state)/2] , state[0] , state[i] , self.species.species[i].mass , self.species.species[i].charge)
        return total_current

    def n_g (self, state) :
        '''total density of neutral gases'''
        total = 0
        for i in(range(len(state)/2)) :
            if self.specie.charge(self.species.species[i]) == 0 :
                total += state[i]
        return total
        
    def solve(self, t0, tf):
        y0 = np.array([self.chamber.T_e_0, self.chamber.T_g_0, self.chamber.n_e_0, self.chamber.n_g_0])
        return solve_ivp(self.f_dy, (t0, tf), y0, method='LSODA')


    def solve_for_I_coil(self, coil_currents):
        """Calculates for a list of intensity in the coil the resulting power consumption and the resulting thrust.
            ## Returns
            power_array , list_of(`state` after long time)"""
        power_array = np.zeros(coil_currents.shape[0])
        final_states = np.zeros((coil_currents.shape[0], 4))  #shape = (y,x)

        for i, I_coil in enumerate(coil_currents):
            self.chamber.I_coil = I_coil

            sol = self.solve(0, 5e-2)    # TODO Needs some testing

            final_state = sol.y[:, -1]

            collision_frequencies = np.zeros(self.species.nb)
            for reac in self.reaction_set:
                if isinstance(reac, GeneralElasticCollision) :
                    sp_idx, freq = reac.colliding_specie_and_collision_frequency(final_state)
                    collision_frequencies[sp_idx] = freq
            eps_p = self.eps_p(collision_frequencies, final_state)

            # calculation of P_abs : the power given by the antenna to the plasma

            power_array[i] = self.P_abs(self.R_ind(eps_p))

            final_states[i] = final_state
            
        return power_array, final_states
