# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 14:41:30 2025

@author: liamg
"""

from scipy.constants import pi, e, k, epsilon_0 as eps_0, c, m_e
from src.reactions.excitation_reaction import Excitation
from src.reactions.ionisation_reaction import Ionisation
from src.reactions.elastic_collision_with_electrons_reaction import ElasticCollisionWithElectron
from src.reactions.flux_to_walls_and_grids_reaction import FluxToWallsAndThroughGrids
from src.reactions.gas_injection_reaction import GasInjection
from src.specie import Species, Specie
from src.constant_rate_calculation import get_K_func

def get_species_and_reactions(chamber):
    
    species_list = Species([Specie("e", m_e, -e), Specie("N_2", 4.65e-26, 0), Specie("N", 2.33e-26, 0), Specie("N_2+", 4.65e-26, e), Specie("N+", 2.33e-26, e), Specie("O_2+", 5.31e-26, e), Specie("O_2", 5.31e-26, 0), Specie("O", 2.67e-26, 0), Specie("O+", 2.67e-26, e)])
    
    exc1_N2 = Excitation(species_list, "N2", get_K_func(species_list, "N2", "exc1_N2"), 6.17, chamber)
    exc2_N2 = Excitation(species_list, "N2", get_K_func(species_list, "N2", "exc2_N2"), 7.35, chamber)
    exc3_N2 = Excitation(species_list, "N2", get_K_func(species_list, "N2", "exc3_N2"), 7.36, chamber)
    exc4_N2 = Excitation(species_list, "N2", get_K_func(species_list, "N2", "exc4_N2"), 8.16, chamber)
    exc5_N2 = Excitation(species_list, "N2", get_K_func(species_list, "N2", "exc5_N2"), 8.40, chamber)
    exc6_N2 = Excitation(species_list, "N2", get_K_func(species_list, "N2", "exc6_N2"), 8.55, chamber)
    exc7_N2 = Excitation(species_list, "N2", get_K_func(species_list, "N2", "exc7_N2"), 8.89, chamber)
    exc8_N2 = Excitation(species_list, "N2", get_K_func(species_list, "N2", "exc8_N2"), 12.50, chamber)
    exc9_N2 = Excitation(species_list, "N2", get_K_func(species_list, "N2", "exc9_N2"), 12.90, chamber)
    exc10_N2 = Excitation(species_list, "N2", get_K_func(species_list, "N2", "exc10_N2"), 12.10, chamber)
    exc11_N2 = Excitation(species_list, "N2", get_K_func(species_list, "N2", "exc11_N2"), 12.90, chamber)
    exc12_N2 = Excitation(species_list, "N2", get_K_func(species_list, "N2", "exc12_N2"), 11.00, chamber)
    exc13_N2 = Excitation(species_list, "N2", get_K_func(species_list, "N2", "exc13_N2"), 11.90, chamber)
    exc14_N2 = Excitation(species_list, "N2", get_K_func(species_list, "N2", "exc14_N2"), 12.30, chamber)
    ion_N2 = Ionisation(species_list, "N2", "N2+", get_K_func(species_list, "N2", "ion_N2"), 15.60, chamber)
    exc1_N = Excitation(species_list, "N", get_K_func(species_list, "N", "exc1_N"), 3.20, chamber)
    exc2_N = Excitation(species_list, "N", get_K_func(species_list, "N", "exc2_N"), 4.00, chamber)
    ion_N = Ionisation(species_list, "N", "N+", get_K_func(species_list, "N", "ion_N"), 14.80, chamber)
    ela_elec_N = ElasticCollisionWithElectron(species_list, "N", get_K_func(species_list, "N", "ela_N"), 0, chamber)
    # dion_N2 = Reaction(species_list, "N2", "N, N+", "N+", "e", get_K_func(species_list, "N2", "dion_N2"), 18.00, [1., 1., 1., 1.]) Hassoul 
    ela_elec_N2 = ElasticCollisionWithElectron(species_list, "N2", get_K_func(species_list, "N2", "ela_N2"), 0, chamber)
    
    exc1_O2 = Excitation(species_list, "O2", "O2", get_K_func(species_list, "O2", "exc1_O2"), 1.00, chamber)
    exc2_O2 = Excitation(species_list, "O2", "O2", get_K_func(species_list, "O2", "exc2_O2"), 1.50, chamber)
    exc3_O2 = Excitation(species_list, "O2", "O2", get_K_func(species_list, "O2", "exc3_O2"), 4.50, chamber)
    exc4_O2 = Excitation(species_list, "O2", "O2", get_K_func(species_list, "O2", "exc4_O2"), 7.10, chamber)
    ion_O2 = Ionisation(species_list, "O2", "O2+", get_K_func(species_list, "O2", "ion_O2"), 12.10, chamber)
    ela_elec_O2 = ElasticCollisionWithElectron(species_list, "O2", get_K_func(species_list, "O2", "ela_elec_O2"), 0, chamber)
    # diss1_O2 = Reaction(species_list, "O2", "O", get_K_func(species_list, "O2", "diss1_O2"), 6.12, [1., 2.])
    # diss2_O2 = Reaction(species_list, ["O2"], ["O"], get_K_func(species_list, "O2", "diss2_O2"), 8.40, [1., 2.])
    exc1_O = Excitation(species_list, "O", get_K_func(species_list, "O", "exc1_O"), 1.97, chamber)
    exc2_O = Excitation(species_list, "O", get_K_func(species_list, "O", "exc2_O"), 4.19, chamber)
    exc3_O = Excitation(species_list, "O", get_K_func(species_list, "O", "exc3_O"), 9.52, chamber)
    exc4_O = Excitation(species_list, "O", get_K_func(species_list, "O", "exc4_O"), 12, chamber)
    exc5_O = Excitation(species_list, "O", get_K_func(species_list, "O", "exc5_O"), 12, chamber)
    exc6_O = Excitation(species_list, "O", get_K_func(species_list, "O", "exc6_O"), 12, chamber)
    exc7_O = Excitation(species_list, "O", get_K_func(species_list, "O", "exc7_O"), 12, chamber)
    exc8_O = Excitation(species_list, "O", get_K_func(species_list, "O", "exc8_O"), 12, chamber)
    exc9_O = Excitation(species_list, "O", get_K_func(species_list, "O", "exc9_O"), 12, chamber)
    ela_elec_O = ElasticCollisionWithElectron(species_list, "O", get_K_func(species_list, "O", "ela_elec_O"), 0, chamber)
    
    # Reaction list
    reaction_list = [
        exc1_N2, exc2_N2, exc3_N2, exc4_N2, exc5_N2, exc6_N2, exc7_N2, exc8_N2, exc9_N2, exc10_N2,
        exc11_N2, exc12_N2, exc13_N2, exc14_N2, ion_N2, exc1_N, exc2_N, ion_N, ela_elec_N,
        ela_elec_N2, exc1_O2, exc2_O2, exc3_O2, exc4_O2, ion_O2, ela_elec_O2, exc1_O, exc2_O,
        exc3_O, exc4_O, exc5_O, exc6_O, exc7_O, exc8_O, exc9_O, ela_elec_O
    ]

    return species, reaction_list
