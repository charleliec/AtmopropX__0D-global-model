import json
import os
import warnings
from pathlib import Path
from typing import Callable, Tuple

import numpy as np 
from numpy.typing import NDArray
from scipy.constants import m_e, e, pi, k, epsilon_0 as eps_0, mu_0, c as c_light   # k is k_B -> Boltzmann constant
from scipy.integrate import trapezoid, solve_ivp, odeint
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

#Local modules
from .util import load_csv, load_cross_section
from .specie import Specie, Species
from .reactions import Reaction, GeneralElasticCollision, ElectronHeatingConstantAbsorbedPower, ElectronHeating
from .chamber_caracteristics import Chamber
from .variable_tracker import VariableTracker


class GlobalModel:

    def __init__(self, species: Species, reaction_set: list[Reaction], chamber: Chamber, electron_heating: ElectronHeating | None = None, simulation_name: str = "test_simu", log_folder_path: str|Path = "./logs"):
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
            self.electron_heating = ElectronHeatingConstantAbsorbedPower(species, 0., 1, chamber)
        else:
            self.electron_heating = electron_heating
        self.electron_heating.set_var_tracker(self.var_tracker) 
        np.set_printoptions(edgeitems=3, infstr='inf',linewidth=200)


    def eval_property(self, func, sol):
        """Calculates a property based on 'state' for all 't'.
            sol must be a np.array with shape (nb_of_t's, dimension_of_state) where each line represents a state"""
        prop = np.zeros(sol.shape[0])   #number of instants
        for i in np.arange(sol.shape[0]):
            prop[i] = func(sol[i])
        return prop       
    
    def f_dy(self, t: float, state: NDArray[np.float64], energy_modifier_func: Callable[[float, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]] | None=None, temp_modifier_func: Callable[[float, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]] | None=None):
        """Returns the derivative of the vector 'state' describing the state of plasma.
            'state' has format : [n_e, n_N2, ..., n_N+, T_e, T_monoato, ..., T_diato]"""
        for idx, var in enumerate(state[:self.species.nb]):
            if var < 0:
                print(f"Warning : Negative density in state at t={t}: {state}")
                state[idx] = 0.0
        for var in state[self.species.nb:]:
            if var < 0:
                return state * 0.
                #raise ValueError(f"Negative temperature in state at t={t}: {state}")

        try:
            densities = state[:self.species.nb]
            temp = state[self.species.nb:]

            dy = np.zeros(state.shape)
            dy_densities = np.zeros(self.species.nb)
            dy_energies = np.zeros(3)

            collision_frequency = 0.0
            for reac in self.reaction_set:
                dy_densities += reac.density_change_rate(state)
                dy_energies += reac.energy_change_rate(state)
                if isinstance(reac, GeneralElasticCollision) :
                    sp, freq = reac.colliding_specie_and_collision_frequency(state)
                    collision_frequency += freq

            self.var_tracker.add_value_to_variable("collision_frequency", collision_frequency)
            # Energy given to the electrons via the coil
            volumic_power_absorbed = self.electron_heating.absorbed_power(state, collision_frequency, t) / self.chamber.V_chamber
            dy_energies[0] += volumic_power_absorbed
            self.var_tracker.add_value_to_variable('p_abs', volumic_power_absorbed)
            dy[:self.species.nb] = dy_densities
            dy[self.species.nb:] = dy_energies
            if energy_modifier_func is not None:
                dy = energy_modifier_func(t, state, dy)
                dy_densities = dy[:self.species.nb]
                dy_energies = dy[self.species.nb:]
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
            # self.var_tracker.add_value_to_variable('total_ion_thrust', self.total_ion_thrust(state))
            # self.var_tracker.add_value_to_variable('total_neutral_thrust', self.total_neutral_thrust(state))
            # self.var_tracker.add_value_to_variable('total_thrust', self.total_thrust(state))
            # self.var_tracker.add_value_to_variable('u_B', self.chamber.u_B(state[self.species.nb],2.18e-25))
            # self.var_tracker.add_value_to_variable('ion_current', self.total_ion_current(state))
            string = ( f"\nt={t:15.9e}"
                + "\n       " + " ".join([f"{val.name:^12}" for val in self.species.species]) + " " +  " ".join([f"{val:^12}" for val in ["Te", "Tmono", "Tdiato"]])
                + "\nstate :" + " ".join([f"{val:12.5e}" for val in state])
                + "\n  dy  :" + " ".join([f"{val:12.5e}" for val in dy]) )
            print(string)
        except Exception as exc:
            print(f"Error in f_dy with state = {state}: \n {exc}")
            raise exc
        dy = np.where(state > 1e-5, dy, 0)
        if temp_modifier_func is not None:
            dy = temp_modifier_func(t, state, dy)
            return dy
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
        for sp in self.species.species[1:]:
            if sp.charge != 0 :
               total_thrust += self.chamber.gamma_ion(state[sp.index], state[self.species.nb], sp.mass) * self.chamber.h_L(self.n_g_tot(state)) * sp.mass * self.chamber.v_beam(sp.mass, sp.charge) * self.chamber.beta_i * self.chamber.S_gridded_wall
        return total_thrust
    
    def total_neutral_thrust(self,state):
        total_thrust = 0
        for sp in self.species.species:
            if sp.charge == 0:
                T_neutral = state[self.species.nb + sp.nb_atoms]
                total_thrust += self.chamber.gamma_neutral(state[sp.index], T_neutral, sp.mass) * sp.mass * self.chamber.S_eff_neutrals() * np.sqrt(8*e*T_neutral/(pi*sp.mass)) 
        return total_thrust
    
    def total_thrust(self,state):
        return self.total_neutral_thrust(state) + self.total_ion_thrust(state)

    def total_ion_current(self , state ) :
        '''Calculates the total amount of ion current toxards the grids'''
        total_current = 0
        for sp in self.species.species[1:]:
            if sp.charge != 0 :
                total_current += e * self.chamber.gamma_ion(state[sp.index], state[self.species.nb], sp.mass) * self.chamber.h_L(self.n_g_tot(state))
        return total_current

    def n_g_tot (self, state) :
        '''total density of neutral gases'''
        total = 0
        #for i in(range(len(state)/2)) :
        for i in range(self.species.nb):
            # if self.species.species[i].charge == 0:
            #     total += state[i]
            total += state[i]
        return total
        
    def solve(self, t0: float, tf: float, initial_state: NDArray[np.float64] | list[float], args: Tuple[Callable[[float, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]] | None, Callable[[float, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]] | None] | None=None):
        """
        Solve the model based on the species, the reaction and the initial state.
        
        Parameters
        ----------
        t0 : float
            Time of the beginning of simulation.
        tf : float
            Time of the end of simulation.
        initial_state : ArrayLike
            Initial state of densities and temperature. Must be in the same order as the species and if there is n species,
            initial_state[n] must be the temperature of the electrons, initial_state[n+1] that of monoatomic species, etc.
        args : Tuple | None 
            Will be passed to f_dy. For now can only contain two functions that will alter the derivatives (to fix a temperature or a density for example).
            Functions must have signature `(time, state, derivative) -> altered_derivative`.
            First one modifies derivative when using energy, second one is used after conversion to temperatures.
        
        Returns
        -------
        Bunch object with the following fields defined:

        t : ndarray, shape (n_points,)
            Time points.
        y : ndarray, shape (n, n_points)
            Values of the solution at `t`.
        sol : `OdeSolution` or None
            Found solution as `OdeSolution` instance; None if `dense_output` was
            set to False.
        t_events : list of ndarray or None
            Contains for each event type a list of arrays at which an event of
            that type event was detected. None if `events` was None.
        y_events : list of ndarray or None
            For each value of `t_events`, the corresponding value of the solution.
            None if `events` was None.
        nfev : int
            Number of evaluations of the right-hand side.
        njev : int
            Number of evaluations of the Jacobian.
        nlu : int
            Number of LU decompositions.
        status : int
            Reason for algorithm termination:

                * -1: Integration step failed.
                *  0: The solver successfully reached the end of `tspan`.
                *  1: A termination event occurred.

        message : string
            Human-readable description of the termination reason.
        success : bool
            True if the solver reached the interval end or a termination event
            occurred (``status >= 0``).
        """
        #y0 = np.array([self.chamber.n_e_0, self.chamber.n_g_0, 0, self.chamber.T_e_0, self.chamber.T_g_0, 0])
        y0 = np.array(initial_state)  #np.array([self.chamber.n_e_0, self.chamber.n_g_0, 0, self.chamber.T_e_0, self.chamber.T_g_0, 0])
        sol = solve_ivp(self.f_dy, (t0, tf), y0, method='LSODA', rtol=1e-2, atol=1e-6, first_step=5e-12, min_step=1e-12, args=args)    # , max_step=1e-7
        #log_file_path=self.simulation_name
        self.var_tracker.save_tracked_variables()
        print("Variables saved")
        return sol


    # def solve_for_I_coil(self, coil_currents, t0, tf, initial_state):
    #     """Calculates for a list of intensity in the coil the resulting power consumption and the resulting thrust.
    #         ## Returns
    #         power_array , list_of(`state` after long time)"""
    #     power_array = np.zeros(len(coil_currents))
    #     final_states = np.zeros((len(coil_currents), self.species.nb+3))  #shape = (y,x)
    #     simulation_name = self.simulation_name

    #     for i, I_coil in enumerate(coil_currents):
    #         self.simulation_name = simulation_name + str(i)
    #         self.var_tracker.update_filename(self.simulation_name+".json")

    #         self.electron_heating.coil_current = I_coil

    #         sol = self.solve(t0, tf, initial_state) 

    #         final_state = sol.y[:, -1]

    #         collision_frequencies = np.zeros(self.species.nb)
    #         for reac in self.reaction_set:
    #             if isinstance(reac, GeneralElasticCollision) :
    #                 sp, freq = reac.colliding_specie_and_collision_frequency(final_state)
    #                 collision_frequencies[sp.index]  += freq

    #         # calculation of P_abs : the power given by the antenna to the plasma

    #         power_array[i] = self.electron_heating.absorbed_power(final_state, collision_frequencies)   #self.P_abs(self.R_ind( eps_p  ))
    #         #power_array[i] = self.electron_heating.power_rf(final_state, collision_frequencies)

    #         final_states[i] = final_state
            
    #     return power_array, final_states
    
    # def solve_for_power_fixed(self, power_list, efficiency_list, t0, tf, initial_state):
    #     """Calculates for a list of power absorbed in the coil the resulting stationary values of different variables.
    #         ## Returns
    #         power_array , list_of(`state` after long time)"""
    #     final_states = np.zeros((len(power_list), self.species.nb+3))  #shape = (y,x)
    #     simulation_name = "all_reactions"

    #     for i, power in enumerate(power_list):
    #         self.simulation_name = simulation_name + str(i)
    #         self.var_tracker.update_filename(self.simulation_name+".json")
    #         self.electron_heating.power_absorbed_value = power * efficiency_list[i]

    #         sol = self.solve(t0, tf, initial_state)    # TODO Needs some testing

    #         final_state = sol.y[:, -1]

    #         collision_frequencies = np.zeros(self.species.nb)
    #         for reac in self.reaction_set:
    #             if isinstance(reac, GeneralElasticCollision) :
    #                 sp, freq = reac.colliding_specie_and_collision_frequency(final_state)
    #                 collision_frequencies[sp.index] = freq
    #         #eps_p = self.eps_p(collision_frequencies, final_state)

    #         final_states[i] = final_state
            
    #     return final_states
    
    # def solve_for_RF_power_fixed(self, power_list, t0, tf, initial_state):
    #     """Calculates for a list of power absorbed in the coil the resulting stationary values of different variables.
    #         ## Returns
    #         power_array , list_of(`state` after long time)"""
    #     final_states = np.zeros((len(power_list), self.species.nb+3))  #shape = (y,x)
    #     simulation_name = self.simulation_name

    #     for i, power in enumerate(power_list):
    #         self.simulation_name = simulation_name + str(i)
    #         self.var_tracker.update_filename(self.simulation_name+".json")
    #         self.electron_heating.power_RF = power

    #         sol = self.solve(t0, tf, initial_state) 

    #         final_state = sol.y[:, -1]

    #         final_states[i] = final_state
            
    #     return final_states
    
    

    
