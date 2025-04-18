# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 14:41:30 2025

@author: liamg
"""

from scipy.constants import pi, e, k, epsilon_0 as eps_0, c, m_e
from src.model_components.reactions.excitation_reaction import Excitation
from src.model_components.reactions.ionisation_reaction import Ionisation
from src.model_components.reactions.dissociation_reaction import Dissociation
from src.model_components.reactions.thermic_diffusion import ThermicDiffusion
from src.model_components.reactions.inelastic_collision import InelasticCollision
from src.model_components.reactions.elastic_collision_with_electrons_reaction import ElasticCollisionWithElectron
from src.model_components.reactions.flux_to_walls_and_grids_reaction import FluxToWallsAndThroughGrids
from src.model_components.reactions.gas_injection_reaction import GasInjection
from src.model_components.reactions.electron_heating_by_coil_reaction import ElectronHeatingConstantAbsorbedPower, ElectronHeatingConstantCurrent

from src.model_components.specie import Species, Specie
from src.model_components.constant_rate_calculation import get_K_func

def get_species_and_reactions(chamber):
    
    species_list = Species([Specie("e", m_e, -e, 0, 3/2), Specie("N2", 4.65e-26, 0, 2, 5/2), Specie("N", 2.33e-26, 0, 1, 3/2), Specie("N2+", 4.65e-26, e, 2, 5/2), Specie("N+", 2.33e-26, e, 1, 3/2), Specie("O2+", 5.31e-26, e, 2, 5/2), Specie("O2", 5.31e-26, 0, 2, 5/2), Specie("O", 2.67e-26, 0, 1, 3/2), Specie("O+", 2.67e-26, e, 1, 3/2)])
    
    initial_state = [1e10, 5e14, 8e13, 1e10, 1e10, 1e10, 2e13, 1e15, 1e10, 1.0, 0.03, 0.03] # [e, N2, N, N2+, N+, O2+, O2, O, O+, T_e, T_monoatomique, T_diatomique]
    #peut-être changer initial_state parce qu'il faut qu'il y ait un nb suffisant d'électrons


#  ██▀ ▀▄▀ ▄▀▀ █ ▀█▀ ▄▀▄ ▀█▀ █ ▄▀▄ █▄ █
#  █▄▄ █ █ ▀▄▄ █  █  █▀█  █  █ ▀▄▀ █ ▀█
# N2
    exc1_N2 = Excitation(species_list, "N2", get_K_func(species_list, "N2", "exc1_N2"), 6.17, chamber)
    exc2_N2 = Excitation(species_list, "N2", get_K_func(species_list, "N2", "exc2_N2"), 7.35, chamber)
    exc3_N2 = Excitation(species_list, "N2", get_K_func(species_list, "N2", "exc3_N2"), 7.36, chamber)
    exc4_N2 = Excitation(species_list, "N2", get_K_func(species_list, "N2", "exc4_N2"), 8.16, chamber)
    exc5_N2 = Excitation(species_list, "N2", get_K_func(species_list, "N2", "exc5_N2"), 8.40, chamber)
    exc6_N2 = Excitation(species_list, "N2", get_K_func(species_list, "N2", "exc6_N2"), 8.55, chamber)
    exc7_N2 = Excitation(species_list, "N2", get_K_func(species_list, "N2", "exc7_N2"), 8.89, chamber)
    exc8_N2 = Excitation(species_list, "N2", get_K_func(species_list, "N2", "exc8_N2"), 12.50, chamber)
    exc9_N2 = Excitation(species_list, "N2", get_K_func(species_list, "N2", "exc9_N2"), 12.90, chamber)
    #exc10_N2 = Excitation(species_list, "N2", get_K_func(species_list, "N2", "exc10_N2"), 12.10, chamber)
    exc11_N2 = Excitation(species_list, "N2", get_K_func(species_list, "N2", "exc11_N2"), 12.90, chamber)
    exc12_N2 = Excitation(species_list, "N2", get_K_func(species_list, "N2", "exc12_N2"), 11.00, chamber)
    exc13_N2 = Excitation(species_list, "N2", get_K_func(species_list, "N2", "exc13_N2"), 11.90, chamber)
    exc14_N2 = Excitation(species_list, "N2", get_K_func(species_list, "N2", "exc14_N2"), 12.30, chamber)
# N
    exc1_N = Excitation(species_list, "N", get_K_func(species_list, "N", "exc1_N"), 3.20, chamber)
    exc2_N = Excitation(species_list, "N", get_K_func(species_list, "N", "exc2_N"), 4.00, chamber)
    #ela_elec_N = ElasticCollisionWithElectron(species_list, "N", get_K_func(species_list, "N", "ela_N"), 0, chamber)
    # dion_N2 = Reaction(species_list, "N2", "N, N+", "N+", "e", get_K_func(species_list, "N2", "dion_N2"), 18.00, [1., 1., 1., 1.]) Hassoul 
    #ela_elec_N2 = ElasticCollisionWithElectron(species_list, "N2", get_K_func(species_list, "N2", "ela_N2"), 0, chamber)
# O2    
    exc1_O2 = Excitation(species_list, "O2", get_K_func(species_list, "O2", "exc1_O2"), 1.00, chamber)
    exc2_O2 = Excitation(species_list, "O2", get_K_func(species_list, "O2", "exc2_O2"), 1.50, chamber)
    exc3_O2 = Excitation(species_list, "O2", get_K_func(species_list, "O2", "exc3_O2"), 4.50, chamber)
    exc4_O2 = Excitation(species_list, "O2", get_K_func(species_list, "O2", "exc4_O2"), 7.10, chamber)
# O
    # exc1_O = Excitation(species_list, "O", get_K_func(species_list, "O", "exc1_O"), 1.97, chamber)
    # exc2_O = Excitation(species_list, "O", get_K_func(species_list, "O", "exc2_O"), 4.19, chamber)
    # exc3_O = Excitation(species_list, "O", get_K_func(species_list, "O", "exc3_O"), 9.52, chamber)
    exc4_O = Excitation(species_list, "O", get_K_func(species_list, "O", "exc4_O"), 12, chamber)
    exc5_O = Excitation(species_list, "O", get_K_func(species_list, "O", "exc5_O"), 12, chamber)
    exc6_O = Excitation(species_list, "O", get_K_func(species_list, "O", "exc6_O"), 12, chamber)
    exc7_O = Excitation(species_list, "O", get_K_func(species_list, "O", "exc7_O"), 12, chamber)
    exc8_O = Excitation(species_list, "O", get_K_func(species_list, "O", "exc8_O"), 12, chamber)
    #exc9_O = Excitation(species_list, "O", get_K_func(species_list, "O", "exc9_O"), 12, chamber)
    
#  █ ▄▀▄ █▄ █ █ ▄▀▀ ▄▀▄ ▀█▀ █ ▄▀▄ █▄ █
#  █ ▀▄▀ █ ▀█ █ ▄██ █▀█  █  █ ▀▄▀ █ ▀█
    ion_N = Ionisation(species_list, "N", "N+", get_K_func(species_list, "N", "ion_N"), 14.80, chamber)
    ion_N2 = Ionisation(species_list, "N2", "N2+", get_K_func(species_list, "N2", "ion_N2"), 15.60, chamber)
    ion_O2 = Ionisation(species_list, "O2", "O2+", get_K_func(species_list, "O2", "ion_O2"), 12.10, chamber)

#  ██▀ █   ▄▀▄ ▄▀▀ ▀█▀ █ ▄▀▀   ▄▀▀ ▄▀▄ █   █   █ ▄▀▀ █ ▄▀▄ █▄ █ ▄▀▀
#  █▄▄ █▄▄ █▀█ ▄██  █  █ ▀▄▄   ▀▄▄ ▀▄▀ █▄▄ █▄▄ █ ▄██ █ ▀▄▀ █ ▀█ ▄██
    ela_O2 = ElasticCollisionWithElectron(species_list, "O2", get_K_func(species_list, "O2", "ela_O2"), 0, chamber)
    #ela_elec_O = ElasticCollisionWithElectron(species_list, "O", get_K_func(species_list, "O", "ela_elec_O"), 0, chamber)
    ela_N = ElasticCollisionWithElectron(species_list, "N", get_K_func(species_list, "N", "ela_N"), 0, chamber)
    ela_O = ElasticCollisionWithElectron(species_list, "O", get_K_func(species_list, "O", "ela_O"), 0, chamber)
    ela_N2 = ElasticCollisionWithElectron(species_list, "N2", get_K_func(species_list, "N2", "ela_N2"), 0, chamber)


#  █▀ █   █ █ ▀▄▀ ██▀ ▄▀▀   ▀█▀ ▄▀▄   ▀█▀ █▄█ ██▀   █   █ ▄▀▄ █   █   ▄▀▀   ▄▀▄ █▄ █ █▀▄   ▀█▀ █▄█ █▀▄ ▄▀▄ █ █ ▄▀  █▄█   ▀█▀ █▄█ ██▀   ▄▀  █▀▄ █ █▀▄ ▄▀▀
#  █▀ █▄▄ ▀▄█ █ █ █▄▄ ▄██    █  ▀▄▀    █  █ █ █▄▄   ▀▄▀▄▀ █▀█ █▄▄ █▄▄ ▄██   █▀█ █ ▀█ █▄▀    █  █ █ █▀▄ ▀▄▀ ▀▄█ ▀▄█ █ █    █  █ █ █▄▄   ▀▄█ █▀▄ █ █▄▀ ▄██
    out_N = FluxToWallsAndThroughGrids(species_list, "N", chamber) #la colliding specie ne sert à rien visiblement


#  ▄▀  ▄▀▄ ▄▀▀   █ █▄ █   █ ██▀ ▄▀▀ ▀█▀ █ ▄▀▄ █▄ █
#  ▀▄█ █▀█ ▄██   █ █ ▀█ ▀▄█ █▄▄ ▀▄▄  █  █ ▀▄▀ █ ▀█
    injection_rates = [] #à revoir
    T_injection = 0.03 #à revoir
    src_gaz = GasInjection(species_list, injection_rates, T_injection, chamber)


#  █ █▄ █ ██▀ █   ▄▀▄ ▄▀▀ ▀█▀ █ ▄▀▀   ▄▀▀ ▄▀▄ █   █   █ ▄▀▀ █ ▄▀▄ █▄ █ ▄▀▀   █   █ █ ▀█▀ █▄█   █ ▄▀▄ █▄ █ ▄▀▀   ▄▀▄ ▄▀▀ ▄▀▀ ██▀ █   ██▀ █▀▄ ▄▀▄ ▀█▀ ██▀ █▀▄   ▀█▀ ▄▀▄   ▀█▀ █▄█ ██▀   █   █ ▄▀▄ █   █   ▄▀▀
#  █ █ ▀█ █▄▄ █▄▄ █▀█ ▄██  █  █ ▀▄▄   ▀▄▄ ▀▄▀ █▄▄ █▄▄ █ ▄██ █ ▀▄▀ █ ▀█ ▄██   ▀▄▀▄▀ █  █  █ █   █ ▀▄▀ █ ▀█ ▄██   █▀█ ▀▄▄ ▀▄▄ █▄▄ █▄▄ █▄▄ █▀▄ █▀█  █  █▄▄ █▄▀    █  ▀▄▀    █  █ █ █▄▄   ▀▄▀▄▀ █▀█ █▄▄ █▄▄ ▄██
    in_gaz = InelasticCollision(species_list, "N", chamber)

#  █▀▄ █ ▄▀▀ ▄▀▀ ▄▀▄ ▄▀▀ █ ▄▀▄ ▀█▀ █ ▄▀▄ █▄ █
#  █▄▀ █ ▄██ ▄██ ▀▄▀ ▀▄▄ █ █▀█  █  █ ▀▄▀ █ ▀█
    diss1_O2 = Dissociation(species_list, ["O2"], ["O"], get_K_func(species_list, "O2", "diss1_O2"), 6.12, [1., 2.], chamber)
    diss2_O2 = Dissociation(species_list, ["O2"], ["O"], get_K_func(species_list, "O2", "diss2_O2"), 8.40, [1., 2.], chamber)
    diss_N2 = Dissociation(species_list, ["N2"], ["N"], get_K_func(species_list, "N2", "diss_N2"), 9.76, [1., 2.], chamber)

#  ▀█▀ █▄█ ██▀ █▀▄ █▄ ▄█ █ ▄▀▀   █▀▄ █ █▀ █▀ █ █ ▄▀▀ █ ▄▀▄ █▄ █  
#   █  █ █ █▄▄ █▀▄ █ ▀ █ █ ▀▄▄   █▄▀ █ █▀ █▀ ▀▄█ ▄██ █ ▀▄▀ █ ▀█  
    #th_O2 = ThermicDiffusion(species_list, "O2", 0.005, 0.03, chamber) #revoir le kappa, on n'a pas besoin de la colliding specie
    # th_N2 = ThermicDiffusion(species_list, "N2", 0.005, 0.03, chamber)
    # th_O = ThermicDiffusion(species_list, "O", 0.005, 0.03, chamber)
    # th_N = ThermicDiffusion(species_list, "N", 0.005, 0.03, chamber)

    #soit mettre le kappa en instance, soit faire une liste de kappa (mais faut l'associer à la bonne espèce)
    
    # Reaction list
    reaction_list = [
        exc1_N2, exc2_N2, exc3_N2, exc4_N2, exc5_N2, exc6_N2, exc7_N2, exc8_N2, exc9_N2, #exc10_N2,
        exc11_N2, exc12_N2, exc13_N2, exc14_N2, ion_N2, exc1_N, #exc2_N, ela_elec_N, ela_elec_N2, ela_elec_O2
        ion_N, exc1_O2, exc2_O2, exc3_O2, exc4_O2, ion_O2, #exc1_O, exc2_O,  exc9_O, ela_elec_O, exc3_O,
        exc4_O, exc5_O, exc6_O, exc7_O, exc8_O
    ]

#  ██▀ █   ██▀ ▄▀▀ ▀█▀ █▀▄ ▄▀▄ █▄ █   █▄█ ██▀ ▄▀▄ ▀█▀ █ █▄ █ ▄▀    ██▄ ▀▄▀   ▀█▀ █▄█ ██▀   ▄▀▀ ▄▀▄ █ █    
#  █▄▄ █▄▄ █▄▄ ▀▄▄  █  █▀▄ ▀▄▀ █ ▀█   █ █ █▄▄ █▀█  █  █ █ ▀█ ▀▄█   █▄█  █     █  █ █ █▄▄   ▀▄▄ ▀▄▀ █ █▄▄  
    electron_heating = ElectronHeatingConstantAbsorbedPower(species_list, 500, 0.6, chamber)

    return species_list, initial_state, reaction_list, electron_heating

