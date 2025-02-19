from typing import override
import numpy as np
from numpy.typing import NDArray
from scipy.constants import m_e, e, pi, k as k_B, epsilon_0 as eps_0, mu_0   # k is k_B -> Boltzmann constant

from src.specie import Specie, Species

from src.reactions.reaction import Reaction

class ElasticCollisionWithElectron(Reaction):
    """
    Elastic collision between a particle and an electron
    Works with 3 temperatures : Te, Tmono, Tdiat
    In reactives, electron must be in first position and colliding_specie next.
    """

    def __init__(self, species: Species, colliding_specie: str, rate_constant, energy_treshold: float):
        """
        Dissociation class
        /!\ Electrons should NOT be added to reactives and products
            Inputs : 
                species : instance of class Species, lists all species present 
                colliding_specie : name of specie that collides with an electron. Must be a string !
                rate_constant : function taking as argument state [n_e, n_N2, ..., n_N+, T_e, T_monoato, ..., T_diato]
                energy_threshold : energy threshold of electron so that reaction occurs
        """
        super().__init__(species, [species.names[0], colliding_specie], [species.names[0], colliding_specie], rate_constant, energy_treshold)

    @override
    def density_change_rate(self, state):
        return np.zeros(self.species.nb)

    
    @override
    def energy_change_rate(self, state):
        rate = np.zeros(3)

        K = self.rate_constant(state)
        reac_speed = K * np.prod(state[self.reactives_indices])
        mass_ratio = m_e / self.reactives[1].mass  # self.reactives[1].mass is mass of colliding_specie
        delta_temp = state[self.species.nb] - state[self.species.nb + self.reactives[1].nb_atoms]  # Te - Tspecie

        energy_change = 3 * mass_ratio * k_B * delta_temp * reac_speed 
        
        rate[0] = -energy_change
        rate[self.reactives[1].nb_atoms] = energy_change #mono / diatomic particles gain energy, electrons lose energy

        return rate

    def get_eps_i(self, state) :
        """ Renvoye la permittivité diélectrique relative due à une réaction de collision ellastique entre un électron et une espèce neutre"""
        # Il faurda définir omega et les constantes physiques : m_e, eps_0, et e
        omega_pe_sq = (n_e * e**2) / (m_e * eps_0)
        # le carré de la pulsation plasma
        for specie in self.reactives :
            # On part du principe qu'il n'y a que deux réactifs : l'électron et l'espèce neutre
            index_neutral = specie.index
            if indice != 0 :
                c_neutral = state[index_neutral]
                nu_m_i = c_neutral * rate_constant
        return 1 - (omega_pe_sq / (omega * (omega -  nu_m_i)))
