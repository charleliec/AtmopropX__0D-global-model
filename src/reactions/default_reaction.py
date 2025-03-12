from typing import override
import numpy as np
from numpy.typing import NDArray
from scipy.constants import m_e, e, pi, k as k_B, epsilon_0 as eps_0, mu_0   # k is k_B -> Boltzmann constant

from src.specie import Specie, Species
from src.reactions.reaction import Reaction


class DefaultReaction(Reaction):
    """
        Represents default reaction of an interaction between an electron and a molecule 
        where reaction speed is K * n_e * n_mol ...
        Works for 3 temperatures : that of electrons, monoatomic and diatomic particles

    """

    def __init__(self, 
                 species: Species, 
                 reactives: list[str], 
                 products: list[str], 
                 rate_constant, 
                 energy_treshold: float, 
                 stoechio_coeffs: list[float]=None, 
                 spectators: list[str]=None 
                 ):
        """
        Reaction class
        /!\ Electrons should be added to reactives and products
            Inputs : 
                species : instance of class Species, lists all species present 
                reactives : list with all reactives names
                products : list with all products names
                rate_constant : function taking as argument state [n_e, n_N2, ..., n_N+, T_e, T_monoato, ..., T_diato]
                energy_threshold : energy threshold of electron so that reaction occurs
                stoechio_coeffs : stoechiometric coefficients always positive
                spectators : list with spectators names (used to print reaction)
        """
        super().__init__(species, reactives, products, stoechio_coeffs, spectators)

        self.energy_threshold = energy_treshold
        self.rate_constant = rate_constant   # func
        self.spectators = spectators
        
    @override
    def density_change_rate(self, state: NDArray[float]): # type: ignore
        """Returns an np.array with the change rate for each species due to this reaction
        state has format : [n_e, n_N2, ..., n_N+, T_e, T_monoato, ..., T_diato]"""
        rate = np.zeros(self.species.nb)

        K = self.rate_constant(state[self.species.nb:])
        product = K * np.prod(state[self.reactives_indices]) # product of rate constant and densities of all the stuff
        for sp in self.reactives:
            i = self.species.get_index_by_instance(sp)
            rate[i] = - product * self.stoechio_coeffs[i]
        for sp in self.products:
            i = self.species.get_index_by_instance(sp)
            rate[i] = + product * self.stoechio_coeffs[i]
        
        return rate

    
    @override
    def energy_change_rate(self, state):
        rate = np.zeros(3)

        K = self.rate_constant(state)
        reac_speed = K * np.prod(state[self.reactives_indices])
        for reac_sp in self.reactives:
            if reac_sp.nb_atoms == 0 :
                electron_power_loss = self.energy_threshold * reac_speed # product of energy, rate constant and densities of all the stuff
                rate[0] = electron_power_loss

            elif reac_sp.nb_atoms == 1 :
                mass_ratio = m_e / reac_sp.mass
                delta_temp = state[self.species.nb] - state[self.species.nb + reac_sp.nb_atoms] # T_e - T_monoatomic
                monoatomic_energy_change = 3 * mass_ratio * k_B * delta_temp * reac_speed
                rate[1] = monoatomic_energy_change

            elif reac_sp.nb_atoms == 2 :
                mass_ratio = m_e / reac_sp.mass
                delta_temp = state[self.species.nb] - state[self.species.nb + reac_sp.nb_atoms] # T_e - T_diatomic
                diatomic_energy_change = 3 * mass_ratio * k_B * delta_temp * reac_speed 
                rate[2] = diatomic_energy_change

            else:
                raise ValueError("Mass number {reac_sp.mass} is not currently handled.")
        return rate

























