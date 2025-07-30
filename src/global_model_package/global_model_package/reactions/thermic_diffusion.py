from typing import override, Callable
import numpy as np
from numpy.typing import NDArray
from scipy.constants import m_e, e, pi, k as k_B, epsilon_0 as eps_0, mu_0   # k is k_B -> Boltzmann constant

from global_model_package.specie import Specie, Species
from .reaction import Reaction
from global_model_package.chamber_caracteristics import Chamber


# * Checked by me

class ThermicDiffusion(Reaction):
    """
        Represents change in temperature due to thermic exchange with the walls.
        Works for 3 temperatures : that of electrons, monoatomic and diatomic particles

    """

    def __init__(self, 
                 species: Species, 
                 kappa : Callable[[float], float],
                 temp_wall : float,
                 chamber: Chamber
                 ):
        """
        Thermic diffusion
        
        Inputs : 
            species : instance of class Species, lists all species present 
            kappa : dict whose keys are species name and values are Functions taking as input the atom temperature and returning the diffusion coefficient
            temp_wall : temperature of the walls
            chamber : instance of the Chamber class with the Chamber caracteristics
        """
        # species.names[0] nom des électrons
        super().__init__(species, species.names, species.names, chamber)
        self.kappa = kappa
        self.temp_wall=temp_wall
        
    @override
    def density_change_rate(self, state: NDArray[np.float64]):
        """Returns an np.array with the change rate for each species due to this reaction
        state has format : [n_e, n_N2, ..., n_N+, T_e, T_monoato, ..., T_diato]"""
        rate = np.zeros(self.species.nb)
        
        return rate

    
    @override
    def energy_change_rate(self, state):
        rate = np.zeros(3)
        lambda_0 = self.chamber.L/2.405 + self.chamber.R/np.pi

        for sp in self.species.species[1:] :   # electron are skipped because handled before
            if sp.charge == 0:
                rate[sp.nb_atoms] -= self.kappa[self.species.names[sp.index]](state[self.species.nb+sp.nb_atoms]) * e * (state[self.species.nb+sp.nb_atoms] - self.temp_wall) * self.chamber.S_total/(k_B *lambda_0*self.chamber.V_chamber)

        self.var_tracker.add_value_to_variable_list('energy_change_thermic_diffusion', rate) # type: ignore

        return rate

























