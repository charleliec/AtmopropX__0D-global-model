from scipy.constants import pi, e, k, epsilon_0 as eps_0, c, m_e

from src.model_components.reactions.excitation_reaction import Excitation
from src.model_components.reactions.ionisation_reaction import Ionisation
from src.model_components.reactions.elastic_collision_with_electrons_reaction import ElasticCollisionWithElectron
from src.model_components.reactions.flux_to_walls_and_grids_reaction import FluxToWallsAndThroughGrids
from src.model_components.reactions.gas_injection_reaction import GasInjection
from src.model_components.reactions.electron_heating_by_coil_reaction import ElectronHeatingConstantAbsorbedPower

from src.model_components.specie import Species, Specie
from src.model_components.constant_rate_calculation import get_K_func


def get_species_and_reactions(chamber):
                    
    species = Species([Specie("e", m_e, -e, 0, 3/2), Specie("Xe", 2.18e-25, 0, 1, 3/2), Specie("Xe+", 2.18e-25, e, 1, 3/2)])

    ### Excitation
    #exc_Xe = Excitation(species, "Xe", get_K_func(species, "Xe", "exc_Xe"), 11.6, chamber) 

    ### Terme source
    src_Xe = GasInjection(species, [0.0, 5e15, 0], 5, chamber) 

    ### Sortie de Xe à travers les grilles
    #out_Xe = FluxToWallsAndThroughGrids(species, "Xe", get_K_func(species, "Xe", "out_Xe"), 0, chamber) 

    # Reaction list
    reaction_list = [src_Xe] #[exc_Xe, src_Xe] #[exc_Xe, src_Xe, out_Xe]

    electron_heating = ElectronHeatingConstantAbsorbedPower(species, 1e3, chamber) 

    return species, reaction_list, electron_heating
