from typing import override
import numpy as np
from numpy.typing import NDArray
from scipy.constants import m_e, e, pi, k as k_B, epsilon_0 as eps_0, mu_0   # k is k_B -> Boltzmann constant

from src.specie import Specie, Species
from src.auxiliary_funcs import *
from src.reactions.reaction import Reaction
from src.chamber_caracteristics import Chamber


# * A check
# ! Valable uniquement pour Chabert
# ! Vérifier les surfaces : |   Je sais pas si j'ai pris les bonnes !!!!!

class FluxToWallsAndThroughGrids(Reaction):
    """
    Flux of electrons, ions and neutrals to the grids OR flux of ions and neutrals to the walls.
    Works with 3 temperatures : Te, Tmono, Tdiat
    In reactives, electron must be in first position and colliding_specie next.
    """

    def __init__(self, species: Species, colliding_specie: str, rate_constant, energy_treshold: float, chamber: Chamber):
        """
        FluxToWallsAndThroughGrids class
            Inputs : 
                species : instance of class Species, lists all species present 
                colliding_specie : name of specie that collides with an electron. Must be a string !
                rate_constant : function taking as argument state [n_e, n_N2, ..., n_N+, T_e, T_monoato, ..., T_diato]
                energy_threshold : energy threshold of electron so that reaction occurs
                chamber : chamber parameters of the chamber in which the reactions are taking place
        """
        super().__init__(species, [species.names[0], colliding_specie], [species.names[0], colliding_specie], chamber)
        self.rate_constant = rate_constant
        self.energy_treshold = energy_treshold

    @override
    def density_change_rate(self, state):
        rate = np.zeros(self.species.nb)
        rate[0] = - self.chamber.gamma_e(state[0], state[self.species.nb]) * self.chamber.S_eff / self.chamber.V_chamber
        for sp in self.species.species[1:] :   # electron are skipped because handled before
            if sp.charge != 0:
                rate[sp.index] = - self.chamber.gamma_ion(state[sp.index], state[self.species.nb + sp.nb_atoms]) * S_ion / V_chamber
            else:
                rate[sp.index] = - self.chamber.gamma_neutral(state[sp.index], state[self.species.nb + sp.nb_atoms]) * S_neutral / V_chamber
        return rate

    
    @override
    def energy_change_rate(self, state):

        rate = np.zeros(3)

        E_kin = 7*e*state[self.species.nb]

        rate[0] = - E_kin * self.chamber.gamma_e(state[0], state[self.species.nb]) * S_eff / V_chamber


        # * Neglected for now because missing energy of ion
        # for sp in self.species.species[1:] :   # electron are skipped because handled before
        #     if sp.charge != 0:
        #         rate[sp.index] = - self.chamber.gamma_ion(state[sp.index], state[self.species.nb + sp.nb_atoms]) * S_ion / V_chamber
        #     else:
        #         rate[sp.index]

        return rate
