from typing import override
import numpy as np
from numpy.typing import NDArray
from scipy.constants import m_e, e, pi, k as k_B, epsilon_0 as eps_0, mu_0   # k is k_B -> Boltzmann constant

from src.model_components.specie import Specie, Species
from src.model_components.reactions.reaction import Reaction
from src.model_components.chamber_caracteristics import Chamber

class Dissociation(Reaction):
    """
    Elastic collision between a particle and an electron
    Works with 3 temperatures : Te, Tmono, Tdiat
    In reactives, electron must be in first position and colliding_specie next.
    """

    def __init__(self, species: Species, rate_constant, energy_threshold: float, chamber: Chamber):
        """ Dissociation class
        Parameters :
        ------------- 
        species : Instance of class Species, lists all species present 
        colliding_specie : Name of specie that collides with an electron. Must be a string !
        rate_constant : Function taking as argument state [n_e, n_N2, ..., n_N+, T_e, T_monoato, ..., T_diato]
        energy_threshold : Energy threshold of electron so that reaction occurs
        """
        super().__init__(species, [species.names[0], species.names], [species.names[0], species.names], chamber)
        self.rate_constant = rate_constant
        self.energy_threshold = energy_threshold

    @override
    def density_change_rate(self, state):
        rate = np.zeros(self.species.nb)
        K = self.rate_constant(state)
        sp_diato = self.reactives[1]
        rate[sp_diato.index] -=  K * np.prod(state[self.reactives_indices])
        for sp in self.species.species[1:]:
            rate[sp.index] += 2 * K * np.prod(state[self.reactives_indices])
        return np.zeros(self.species.nb)
    
    @override
    def energy_change_rate(self, state):
        rate = np.zeros(3)

        K = self.rate_constant(state)
        reac_speed = K * np.prod(state[self.reactives_indices])
        
        rate[0] =  reac_speed * self.energy_threshold
        
        Delta_E = 0.5 #d'après Esteves

        for sp in self.species.species:
            if sp.nb_atoms == 1:
                rate[sp.nb_atoms] += reac_speed * 2 * Delta_E 
        
        return rate