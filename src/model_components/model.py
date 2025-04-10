import numpy as np 
from numpy.typing import NDArray
from scipy.constants import m_e, e, pi, k, epsilon_0 as eps_0, mu_0, c as c_light   # k is k_B -> Boltzmann constant
from scipy.integrate import trapezoid, solve_ivp, odeint
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import json
import os
import warnings

#Local modules
from src.model_components.util import load_csv, load_cross_section
# from src.auxiliary_funcs import pressure, maxwellian_flux_speed, u_B, A_eff, A_eff_1, SIGMA_I, h_L
from src.model_components.specie import Specie, Species
from src.model_components.reactions.reaction import Reaction
from src.model_components.reactions.general_elastic_collision import GeneralElasticCollision
from src.model_components.reactions.electron_heating_by_coil_reaction import ElectronHeatingConstantAbsorbedPower, ElectronHeating
from src.model_components.chamber_caracteristics import Chamber
from src.model_components.variable_tracker import VariableTracker


class GlobalModel:

    def __init__(self, species: Species, reaction_set: list[Reaction], chamber: Chamber, electron_heating: ElectronHeating = None, simulation_name: str = "test_simu", log_folder_path: str = "./logs"):
        """Object simulating the evolution of a plasma with 0D model. 
            Inputs :
                config_dict : dictionary containing all parameters about the experimental setup
                species : instance of Species with all species being considered
                reaction_set : list with all reactions being considered -> [Reaction]"""
        self.species = species
        self.reaction_set = reaction_set
        self.chamber = chamber
        self.simulation_name = simulation_name
        self.var_tracker = VariableTracker(log_folder_path, simulation_name+".json")

        for reac in reaction_set:
            reac.set_var_tracker(self.var_tracker)

        if electron_heating is None:
            warnings.warn("No electron heating reaction was provided")
            self.electron_heating = ElectronHeatingConstantAbsorbedPower(species, 0., chamber)
        else:
            self.electron_heating = electron_heating
        self.electron_heating.set_var_tracker(self.var_tracker) 


    def eval_property(self, func, sol):
        """Calculates a property based on 'state' for all 't'.
            sol must be a np.array with shape (nb_of_t's, dimension_of_state) where each line represents a state"""
        prop = np.zeros(sol.shape[0])   #number of instants
        for i in np.arange(sol.shape[0]):
            prop[i] = func(sol[i])
        return prop       
    
    def f_dy(self, t, state):
        """Returns the derivative of the vector 'state' describing the state of plasma.
            'state' has format : [n_e, n_N2, ..., n_N+, T_e, T_monoato, ..., T_diato]"""
        
        try:
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
                    sp, freq = reac.colliding_specie_and_collision_frequency(state)
                    collision_frequencies[sp.index] += freq

            self.var_tracker.add_value_to_variable_list("dy_energy_", dy_energies, "_before_heating")
            # Energy given to the electrons via the coil
            dy_energies[0] += self.electron_heating.absorbed_power(state, collision_frequencies) / self.chamber.V_chamber
            self.var_tracker.add_value_to_variable('power', self.electron_heating.absorbed_power(state, collision_frequencies) / self.chamber.V_chamber)

            self.var_tracker.add_value_to_variable_list("dy_energy_", dy_energies, "_atom")
            # total thermal capacity (in Joule / eV ) of all species with same number of atoms : sum of (3/2 or 5/2 * e * density)
            # here E = total_thermal_capacity * T (in eV)
            total_thermal_capacity_by_sp_type = np.zeros(3)
            dy_total_thermal_capacity_by_sp_type = np.zeros(3)
            for sp in self.species.species :
                total_thermal_capacity_by_sp_type[sp.nb_atoms] += sp.thermal_capacity * e * densities[sp.index]
                dy_total_thermal_capacity_by_sp_type[sp.nb_atoms] += sp.thermal_capacity * e * dy_densities[sp.index]
            
            #Transform derivative of energy into derivative of temperature
            dy_temp = (dy_energies - temp * dy_total_thermal_capacity_by_sp_type) / total_thermal_capacity_by_sp_type

            dy[:self.species.nb] = dy_densities
            dy[self.species.nb:] = np.nan_to_num(dy_temp, nan=0.0)

            self.var_tracker.add_value_to_variable("time", t)
            self.var_tracker.add_all_densities_and_temperatures(state, self.species)
            self.var_tracker.add_all_densities_and_temperatures(dy, self.species, prefix="dy_")
            energies = total_thermal_capacity_by_sp_type * temp
            self.var_tracker.add_value_to_variable_list("energy_", energies, "_atom")
            self.var_tracker.add_value_to_variable('h_L', self.chamber.h_L(self.n_g_tot(state)))
            self.var_tracker.add_value_to_variable('h_R', self.chamber.h_R(self.n_g_tot(state)))
            print(f"Derivative calculated for state at t={t}: {state}")
        except Exception as exc:
            print(f"Error in f_dy: {exc}")
            raise e

        #self.var_tracker.add_value_to_variable("collision_frequencies", collision_frequencies)
        return dy
    
# TODO a coder
    # def thrust_i(self, T_e, n_e, n_ion , m_ion , charge): #Faux, qui est specie ?
    #     """Thrust produced by the ion beam of one specie"""
    #     return self.chamber.gamma_ion(n_ion, T_e , m_ion, self.n_g_tot(state)) * specie.mass * self.chamber.v_beam(m_ion , charge) * self.chamber.beta_i * pi * self.chamber.R ** 2

    # def j_i(self, T_e, n_e, n_ion , m_ion , charge):
    #     """Ion current density of one ionic specie extracted by the grids"""
    #     return self.chamber.gamma_ion( n_ion, T_e, m_ion) * e * charge

        
    def total_ion_thrust(self , state ) :
        '''Calculates the total amount of thrust generated'''
        total_thrust = 0
        for i in(range(1,len(state)/2)):
            if self.species.species[i].charge != 0 :
                total_thrust += self.thrust_i( state[len(state)/2] , state[0] , state[i] , self.species.species[i].mass , self.species.species[i].charge)


    def total_ion_current(self , state ) :
        '''Calculates the total amount of ion current toxards the grids'''
        total_current = 0
        for i in(range(1,len(state)/2)):
            if self.species.species[i].charge != 0 :
                total_current += self.j_i( state[len(state)/2] , state[0] , state[i] , self.species.species[i].mass , self.species.species[i].charge)
        return total_current

    def n_g_tot (self, state) :
        '''total density of neutral gases'''
        total = 0
        #for i in(range(len(state)/2)) :
        for i in range(self.species.nb):
            #if self.specie.charge(self.species.species[i]) == 0 :
            if self.species.species[i].charge == 0:
                total += state[i]
        return total
        
    def solve(self, t0, tf):
        #y0 = np.array([self.chamber.n_e_0, self.chamber.n_g_0, 0, self.chamber.T_e_0, self.chamber.T_g_0, 0])
        y0 = np.array([self.chamber.n_e_0, self.chamber.n_g_0, self.chamber.n_Xe_plus, self.chamber.T_e_0, self.chamber.T_g_0, 0])
        sol = solve_ivp(self.f_dy, (t0, tf), y0, method='LSODA', rtol=1e-8, atol=1e-15, first_step=5e-13, min_step=1e-15)    # , max_step=1e-7
        #log_file_path=self.simulation_name
        self.var_tracker.save_tracked_variables()
        return sol


    def solve_for_I_coil(self, coil_currents, tf = 5e-2):
        """Calculates for a list of intensity in the coil the resulting power consumption and the resulting thrust.
            ## Returns
            power_array , list_of(`state` after long time)"""
        power_array = np.zeros(coil_currents.shape[0])
        final_states = np.zeros((coil_currents.shape[0], self.species.nb+3))  #shape = (y,x)

        for i, I_coil in enumerate(coil_currents):
            self.chamber.I_coil = I_coil

            sol = self.solve(0, tf)    # TODO Needs some testing

            final_state = sol.y[:, -1]

            collision_frequencies = np.zeros(self.species.nb)
            for reac in self.reaction_set:
                if isinstance(reac, GeneralElasticCollision) :
                    sp, freq = reac.colliding_specie_and_collision_frequency(final_state)
                    collision_frequencies[sp.index] = freq
            eps_p = self.eps_p(collision_frequencies, final_state)

            # calculation of P_abs : the power given by the antenna to the plasma

            power_array[i] = self.electron_heating.absorbed_power(final_state, collision_frequencies)   #self.P_abs(self.R_ind( eps_p  ))

            final_states[i] = final_state
            
        return power_array, final_states
    
    def solve_for_power_fixed(self, power_list, tf = 1):
        """Calculates for a list of power absorbed in the coil the resulting stationary values of different variables.
            ## Returns
            power_array , list_of(`state` after long time)"""
        final_states = np.zeros((len(power_list), self.species.nb+3))  #shape = (y,x)
        simulation_name = "all_reactions"

        for i, power in enumerate(power_list):
            self.simulation_name = simulation_name + str(i)
            self.var_tracker = VariableTracker("./logs", self.simulation_name+".json")
            self.electron_heating.power_absorbed_value=power

            sol = self.solve(0, tf)    # TODO Needs some testing

            final_state = sol.y[:, -1]

            collision_frequencies = np.zeros(self.species.nb)
            for reac in self.reaction_set:
                if isinstance(reac, GeneralElasticCollision) :
                    sp, freq = reac.colliding_specie_and_collision_frequency(final_state)
                    collision_frequencies[sp.index] = freq
            #eps_p = self.eps_p(collision_frequencies, final_state)

            final_states[i] = final_state
            
        return final_states
    

    
