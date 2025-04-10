from typing import override
import numpy as np
from numpy.typing import NDArray
from scipy.constants import m_e, e, pi, k as k_B, epsilon_0 as eps_0, mu_0   # k is k_B -> Boltzmann constant

from src.model_components.specie import Specie, Species
from src.model_components.reactions.reaction import Reaction
from src.model_components.chamber_caracteristics import Chamber


# * OK ! Vérif par Liam
# ! A recheck pour Chamber

class GasInjection(Reaction):
    """
    Represents the gas injection : adds quantity to ne, ...
    """

    def __init__(self, species: Species, injection_rates: NDArray[np.float64], T_injection: float, chamber: Chamber):
        """
        Dissociation class
            Inputs : 
                species : instance of class Species, lists all species present 
                injection_rate : np.array with number of particle injected per second for each specie
                    Must have same dimesnsions and be in the same order as species.names
                T_injection : temperature at which the gas is injected into the chamber
        """
        species_injected_names = [species.names[i] for i in range(species.nb) if injection_rates[i] != 0]
        super().__init__(species, [], species_injected_names, chamber)
        self.injection_rates = np.array(injection_rates)
        self.T_injection = T_injection

    @override
    def density_change_rate(self, state):
        rate = self.injection_rates / self.chamber.V_chamber
        self.var_tracker.add_value_to_variable_list("density_change_gas_injection", rate)
        return rate

    
    @override
    def energy_change_rate(self, state):
        rate = np.zeros(3)

        for sp in self.products:
            rate[sp.nb_atoms] += 3/2 * self.injection_rates[sp.index] * e * self.T_injection / self.chamber.V_chamber
        self.var_tracker.add_value_to_variable_list("energy_change_gas_injection", rate)
        return rate