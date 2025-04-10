from typing import override
import numpy as np
from numpy.typing import NDArray
from scipy.constants import m_e, e, pi, k as k_B, epsilon_0 as eps_0, mu_0   # k is k_B -> Boltzmann constant

from src.model_components.specie import Specie, Species
from src.model_components.reactions.reaction import Reaction
from src.model_components.chamber_caracteristics import Chamber


# * Checked by me

class Ionisation(Reaction):
    """
        Represents ionisation of a molecule by an electron
        where reaction speed is K * n_e * n_mol ...
        Works for 3 temperatures : that of electrons, monoatomic and diatomic particles

    """

    def __init__(self, 
                 species: Species, 
                 molecule_before_ionization_name: str, 
                 molecule_after_ionization_name: str,
                 rate_constant, 
                 threshold_energy: float,
                 chamber: Chamber
                 ):
        """
        Reaction class
        
        Inputs : 
            species : instance of class Species, lists all species present 
            molecule_name : name of molecule involved
            rate_constant : function taking as argument state [n_e, n_N2, ..., n_N+, T_e, T_monoato, ..., T_diato]
            energy_threshold : energy threshold of electron so that reaction occurs
        """
        # species.names[0] nom des électrons
        super().__init__(species, [species.names[0], molecule_before_ionization_name], [species.names[0], molecule_after_ionization_name], chamber)

        self.threshold_energy = threshold_energy
        self.rate_constant = rate_constant   # func
        
    @override
    def density_change_rate(self, state: NDArray[float]): # type: ignore
        """Returns an np.array with the change rate for each species due to this reaction
        state has format : [n_e, n_N2, ..., n_N+, T_e, T_monoato, ..., T_diato]"""
        rate = np.zeros(self.species.nb)
        reaction_speed = self.rate_constant(state) * np.prod(state[self.reactives_indices])
        rate[0] = reaction_speed
        # change for molecule before ionization
        rate[self.reactives_indices[1]] = - reaction_speed
        # change for molecule after ionization
        rate[self.products_indices[1]] = reaction_speed
        self.var_tracker.add_value_to_variable('Kiz', self.rate_constant(state))
        self.var_tracker.add_value_to_variable('reaction_speed', reaction_speed)
        self.var_tracker.add_value_to_variable_list("density_change_ionisation", rate)
        
        return rate

    
    @override
    def energy_change_rate(self, state):
        rate = np.zeros(3)

        K = self.rate_constant(state)
        rate[0] -= e*self.threshold_energy * K * np.prod(state[self.reactives_indices])
        self.var_tracker.add_value_to_variable_list('energy_change_ionisation', rate)
        
        return rate

























