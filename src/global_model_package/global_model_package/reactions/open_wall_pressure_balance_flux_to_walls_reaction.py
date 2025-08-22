from typing import override
import numpy as np
from numpy.typing import NDArray
from scipy.constants import m_e, e, pi, k as k_B, epsilon_0 as eps_0, mu_0, N_A  # k is k_B -> Boltzmann constant

from global_model_package.specie import Specie, Species
from .reaction import Reaction
from global_model_package.chamber_caracteristics import Chamber


# * A check
# ! Valable uniquement pour Chabert
# ! Vérifier les surfaces : |   Je sais pas si j'ai pris les bonnes !!!!!

class OpenWallPressureBalanceFluxToWalls(Reaction):
    """
    Maintains a constant pressure in chamber and simulates the flux to the walls. Used to reproduced the experiments of Kim et al. 
    Do not use in the context of a plasma thruster !
    """

    def __init__(self, species: Species, chamber: Chamber):
        """
        OpenWallPressureBalanceFluxToWalls class
            Inputs : 
                species : instance of class Species, lists all species present 
                chamber : chamber parameters of the chamber in which the reactions are taking place.
                            The 'target_pressure' variable MUST be specified here. 
                            Therefore it has to be present in the config.py file.
        """
        super().__init__(species, [], species.names, chamber)
        # WARNING : take sufficiently small value, otherwise delta_pressure will stay high even after a long time (since escape is proportional to delta pressure)
        self.tau = 0.0001  # characteristic time to reach target pressure in seconds
        #self.rate_constant = rate_constant
        #self.energy_treshold = energy_treshold

    def n_g_tot (self, state) :
        '''total density of neutral gases'''
        total = 0
        #for i in(range(len(state)/2)) :
        for i in range(self.species.nb):
            # if self.species.species[i].charge == 0:
            #     total += state[i]
            total += state[i]
        return total

    @override
    def density_change_rate(self, state):
        n_g_tot = self.n_g_tot(state)
        V_corrected = (1+self.chamber.h_R(n_g_tot))*(2+self.chamber.h_L(n_g_tot)) * self.chamber.V_chamber / 6
        rate = np.zeros(self.species.nb)

        pressure = 0
        for sp in self.species.species[1:]:  # electrons do not contrinute to temperature
            pressure += state[sp.index] * 8.31 * (state[self.species.nb + sp.nb_atoms] * e / k_B) / N_A #in Pa
        delta_pressure = max(pressure - self.chamber.target_pressure, 0)
        self.var_tracker.add_value_to_variable("delta_pressure", delta_pressure)
        self.var_tracker.add_value_to_variable("pressure", pressure)


        total_electron_density_change_rate = 0
        for sp in self.species.species[1:] :   # electron are skipped because handled afterwards
            # escape through the outlet 
            delta_density = delta_pressure * N_A / (8.31 * (state[self.species.nb + sp.nb_atoms] * e / k_B) ) # max temperature limitated
            escape_rate = (state[sp.index] / self.n_g_tot(state)) * delta_density / self.tau
            self.var_tracker.add_value_to_variable(f"escape_rate_{sp.name}", escape_rate) # denominator is the caracteristic time to reach target pressure in seconds
            rate[sp.index] -= escape_rate

            if sp.charge != 0: #neutralization of charged particles to the walls
                gamma_ion = self.chamber.gamma_ion(state[sp.index], state[self.species.nb] , sp.mass)
                neutralization_rate = 7*gamma_ion * self.chamber.S_eff_total(self.n_g_tot(state)) / V_corrected
                self.var_tracker.add_value_to_variable(f"neutralization_rate_{sp.name}", neutralization_rate)

                rate[sp.index] -= neutralization_rate
                neutralized_sp = self.species.get_specie_by_name(sp.name[:-1]) # remove last caracter from string, i.e. "Xe+" becomes "Xe"
                rate[neutralized_sp.index] +=  neutralization_rate

                # escape through open wall with diffusion
                diffusion_flux = 0
                diffusion_flux = self.chamber.gamma_neutral(state[sp.index], state[self.species.nb + sp.nb_atoms], sp.mass) * self.chamber.S_gridded_wall / V_corrected
                rate[sp.index] -= diffusion_flux
                rate[neutralized_sp.index] += diffusion_flux # flux d'atome sortant = entrant

                total_electron_density_change_rate += neutralization_rate + escape_rate + diffusion_flux


        # electron density change rate
        rate[0] -= total_electron_density_change_rate
        self.var_tracker.add_value_to_variable_list("density_change_flux_to_walls_and_through_grids", rate) # type: ignore
        return rate

    
    @override
    def energy_change_rate(self, state):
        n_g_tot = self.n_g_tot(state)
        V_corrected = (1+self.chamber.h_R(n_g_tot))*(2+self.chamber.h_L(n_g_tot)) * self.chamber.V_chamber / 6
        rate = np.zeros(3)

        E_kin = 7*e*state[self.species.nb]
        pressure = 0
        for sp in self.species.species[1:]:  # electrons do not contrinute to temperature
            pressure += state[sp.index] * 8.31 * (state[self.species.nb + sp.nb_atoms] * e / k_B) / N_A #in Pa
        delta_pressure = max(pressure - self.chamber.target_pressure, 0)
    
        #* energy loss for ions neglected for now because missing energy of ion
        gamma_e = 0
        total_diffusion_flux = 0
        for sp in self.species.species[1:] :   # electron are skipped because handled before
            if sp.charge != 0:
                gamma_e += self.chamber.gamma_ion(state[sp.index], state[self.species.nb] , sp.mass)
                #rate[sp.nb_atoms] -= self.chamber.gamma_ion(state[sp.index], state[self.species.nb] , sp.mass) * self.chamber.S_eff_total_ion_neutrelisation(n_g) / self.chamber.V_chamber
                # escape through open wall with diffusion no consideration of energy of ions leaving
                diffusion_flux = self.chamber.gamma_neutral(state[sp.index], state[self.species.nb + sp.nb_atoms], sp.mass) * self.chamber.S_gridded_wall / V_corrected
                #rate[sp.index] -= diffusion_flux
                total_diffusion_flux += diffusion_flux
            
            E_neutral = sp.thermal_capacity * e * state[self.species.nb + sp.nb_atoms]
            delta_density = delta_pressure * N_A / (8.31 * (state[self.species.nb + sp.nb_atoms] * e / k_B)) 
            rate[sp.nb_atoms] -= E_neutral * (state[sp.index] / self.n_g_tot(state)) * delta_density / self.tau
        
        rate[0] -= E_kin * (gamma_e * self.chamber.S_eff_total(self.n_g_tot(state)) / V_corrected + total_diffusion_flux)
        self.var_tracker.add_value_to_variable_list("energy_change_flux_to_walls_and_through_grids", rate) # type: ignore
        return rate
    
    #problème avec les énergies : pour les ions, jsp mais pour les neutres: on prend l'agitation thermique
    #il faut distinguer mono et diato pour les capacités thermiques
