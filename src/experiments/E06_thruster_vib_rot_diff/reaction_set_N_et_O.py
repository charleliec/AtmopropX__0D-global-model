import numpy as np
from scipy.constants import pi, e, k as k_B, epsilon_0 as eps_0, c, m_e
# import numpy as np

from global_model_package.reactions import (Excitation, Ionisation, Dissociation, 
                VibrationalExcitation, RotationalExcitation,
                ThermicDiffusion, InelasticCollision, ElasticCollisionWithElectron, 
                FluxToWallsAndThroughGrids, GasInjection,
                ElectronHeatingConstantRFPower
            )

from global_model_package.specie import Species, Specie
from global_model_package.constant_rate_calculation import get_K_func, ReactionRateConstant

ReactionRateConstant.CROSS_SECTIONS_PATH = "../../../cross_sections"

def get_species_and_reactions(chamber):
    
    species = Species([Specie("e", m_e, -e, 0, 3/2), Specie("N2", 4.65e-26, 0, 2, 5/2), Specie("N", 2.33e-26, 0, 1, 3/2), Specie("N2+", 4.65e-26, e, 2, 5/2), Specie("N+", 2.33e-26, e, 1, 3/2), Specie("O2+", 5.31e-26, e, 2, 5/2), Specie("O2", 5.31e-26, 0, 2, 5/2), Specie("O", 2.67e-26, 0, 1, 3/2), Specie("O+", 2.67e-26, e, 1, 3/2)])

    initial_state_dict = {
        "e": 2.1e12,
        "N2": 8e14,
        "N": 8e14,
        "N2+": 1e12,
        "N+": 1e11,
        "O2+": 0.5e12,
        "O2": 2.5e13,
        "O": 1e15,
        "O+": 0.5e12,
        "T_e": 1.0,
        "T_mono": 0.03,
        "T_diato": 0.03
    }
    compression_rate = 4_000
    initial_state = [compression_rate * initial_state_dict[specie.name] for specie in species.species] + [initial_state_dict["T_e"], initial_state_dict["T_mono"], initial_state_dict["T_diato"]]
    injection_rates = np.array([0.0, 3.2e18, 3.2e16, 0.0, 0.0, 0.0, 9.7e16, 4.3e18, 0.0])
    # initial_state = [3.07635e+09,  1.14872e+15,  5.71817e+13,  1.62203e+03,  1.14818e+03,  1.73333e+03,  4.91217e+13,  7.59081e+14,  1.22910e+03,  1.59358e+10,  1.08048e-01,  3.00124e-02]
    # initial_state = [1e15, 5e14, 8e13, 1e10, 1e10, 1e10, 2e13, 1e15, 1e10, 4.0, 0.03, 0.03] # [e, N2, N, N2+, N+, O2+, O2, O, O+, T_e, T_monoatomique, T_diatomique]
    #peut-être changer initial_state parce qu'il faut qu'il y ait un nb suffisant d'électrons


#  ██▀ ▀▄▀ ▄▀▀ █ ▀█▀ ▄▀▄ ▀█▀ █ ▄▀▄ █▄ █
#  █▄▄ █ █ ▀▄▄ █  █  █▀█  █  █ ▀▄▀ █ ▀█
# N2
    exc1_N2 = Excitation(species, "N2", get_K_func(species, "N2", "exc1_N2"), 6.17, chamber)
    exc2_N2 = Excitation(species, "N2", get_K_func(species, "N2", "exc2_N2"), 7.35, chamber)
    exc3_N2 = Excitation(species, "N2", get_K_func(species, "N2", "exc3_N2"), 7.36, chamber)
    exc4_N2 = Excitation(species, "N2", get_K_func(species, "N2", "exc4_N2"), 8.16, chamber)
    exc5_N2 = Excitation(species, "N2", get_K_func(species, "N2", "exc5_N2"), 8.40, chamber)
    exc6_N2 = Excitation(species, "N2", get_K_func(species, "N2", "exc6_N2"), 8.55, chamber)
    exc7_N2 = Excitation(species, "N2", get_K_func(species, "N2", "exc7_N2"), 8.89, chamber)
    exc8_N2 = Excitation(species, "N2", get_K_func(species, "N2", "exc8_N2"), 12.50, chamber)
    exc9_N2 = Excitation(species, "N2", get_K_func(species, "N2", "exc9_N2"), 12.90, chamber)
    # -- N'existe pas a priori exc10_N2 = Excitation(species_list, "N2", get_K_func(species_list, "N2", "exc10_N2"), 12.10, chamber)
    exc11_N2 = Excitation(species, "N2", get_K_func(species, "N2", "exc11_N2"), 12.90, chamber)
    exc12_N2 = Excitation(species, "N2", get_K_func(species, "N2", "exc12_N2"), 11.00, chamber)
    exc13_N2 = Excitation(species, "N2", get_K_func(species, "N2", "exc13_N2"), 11.90, chamber)
    exc14_N2 = Excitation(species, "N2", get_K_func(species, "N2", "exc14_N2"), 12.30, chamber)
# N
    exc1_N = Excitation(species, "N", get_K_func(species, "N", "exc1_N"), 3.20, chamber)
    exc2_N = Excitation(species, "N", get_K_func(species, "N", "exc2_N"), 4.00, chamber)
    # dion_N2 = Reaction(species_list, "N2", "N, N+", "N+", "e", get_K_func(species_list, "N2", "dion_N2"), 18.00, [1., 1., 1., 1.]) Hassoul 
# O2    
    exc1_O2 = Excitation(species, "O2", get_K_func(species, "O2", "exc1_O2"), 1.00, chamber)
    exc2_O2 = Excitation(species, "O2", get_K_func(species, "O2", "exc2_O2"), 1.50, chamber)
    exc3_O2 = Excitation(species, "O2", get_K_func(species, "O2", "exc3_O2"), 4.50, chamber)
    exc4_O2 = Excitation(species, "O2", get_K_func(species, "O2", "exc4_O2"), 7.10, chamber)
# O
    exc1_O = Excitation(species, "O", get_K_func(species, "O", "exc1_O"), 1.97, chamber)
    exc2_O = Excitation(species, "O", get_K_func(species, "O", "exc2_O"), 4.19, chamber)
    exc3_O = Excitation(species, "O", get_K_func(species, "O", "exc3_O"), 9.52, chamber)
    exc4_O = Excitation(species, "O", get_K_func(species, "O", "exc4_O"), 12, chamber)
    exc5_O = Excitation(species, "O", get_K_func(species, "O", "exc5_O"), 12, chamber)
    exc6_O = Excitation(species, "O", get_K_func(species, "O", "exc6_O"), 12, chamber)
    exc7_O = Excitation(species, "O", get_K_func(species, "O", "exc7_O"), 12, chamber)
    exc8_O = Excitation(species, "O", get_K_func(species, "O", "exc8_O"), 12, chamber)
    exc9_O = Excitation(species, "O", get_K_func(species, "O", "exc9_O"), 12, chamber)
    
#  █ ▄▀▄ █▄ █ █ ▄▀▀ ▄▀▄ ▀█▀ █ ▄▀▄ █▄ █
#  █ ▀▄▀ █ ▀█ █ ▄██ █▀█  █  █ ▀▄▀ █ ▀█
    ion_N = Ionisation(species, "N", "N+", get_K_func(species, "N", "ion_N"), 14.80, chamber)
    ion_N2 = Ionisation(species, "N2", "N2+", get_K_func(species, "N2", "ion_N2"), 15.60, chamber)
    ion_O2 = Ionisation(species, "O2", "O2+", get_K_func(species, "O2", "ion_O2"), 12.10, chamber)
    ion_O = Ionisation(species, "O", "O+", get_K_func(species,"O", "ion_O"), 13.60, chamber)

#  ██▀ █   ▄▀▄ ▄▀▀ ▀█▀ █ ▄▀▀   ▄▀▀ ▄▀▄ █   █   █ ▄▀▀ █ ▄▀▄ █▄ █ ▄▀▀
#  █▄▄ █▄▄ █▀█ ▄██  █  █ ▀▄▄   ▀▄▄ ▀▄▀ █▄▄ █▄▄ █ ▄██ █ ▀▄▀ █ ▀█ ▄██  # * complete
    ela_O2 = ElasticCollisionWithElectron(species, "O2", get_K_func(species, "O2", "ela_O2"), chamber)
    ela_N = ElasticCollisionWithElectron(species, "N", get_K_func(species, "N", "ela_N"), chamber)
    ela_O = ElasticCollisionWithElectron(species, "O", get_K_func(species, "O", "ela_O"), chamber)
    ela_N2 = ElasticCollisionWithElectron(species, "N2", get_K_func(species, "N2", "ela_N2"), chamber)


#  █ █ █ ██▄ █▀▄ ▄▀▄ ▀█▀ █ ▄▀▄ █▄ █ ▄▀▄ █     ██▀ ▀▄▀ ▄▀▀ █ ▀█▀ ▄▀▄ ▀█▀ █ ▄▀▄ █▄ █
#  ▀▄▀ █ █▄█ █▀▄ █▀█  █  █ ▀▄▀ █ ▀█ █▀█ █▄▄   █▄▄ █ █ ▀▄▄ █  █  █▀█  █  █ ▀▄▀ █ ▀█
    vib_exc_N2_list = VibrationalExcitation.from_concatenated_txt_file(species, "N2", "vib_exc", "EXCITATION", chamber)
    vib_exc_O2_list = VibrationalExcitation.from_concatenated_txt_file(species, "O2", "vib_exc", "EXCITATION", chamber)


#  █▀▄ ▄▀▄ ▀█▀ ▄▀▄ ▀█▀ █ ▄▀▄ █▄ █ ▄▀▄ █     ██▀ ▀▄▀ ▄▀▀ █ ▀█▀ ▄▀▄ ▀█▀ █ ▄▀▄ █▄ █
#  █▀▄ ▀▄▀  █  █▀█  █  █ ▀▄▀ █ ▀█ █▀█ █▄▄   █▄▄ █ █ ▀▄▄ █  █  █▀█  █  █ ▀▄▀ █ ▀█
    rot_exc_N2_list = RotationalExcitation.from_concatenated_txt_file(species, "N2", "rot_exc", "ROTATIONAL", chamber)
    rot_exc_O2_list = RotationalExcitation.from_concatenated_txt_file(species, "O2", "rot_exc", "ROTATIONAL", chamber)



#  █▀ █   █ █ ▀▄▀ ██▀ ▄▀▀   ▀█▀ ▄▀▄   ▀█▀ █▄█ ██▀   █   █ ▄▀▄ █   █   ▄▀▀   ▄▀▄ █▄ █ █▀▄   ▀█▀ █▄█ █▀▄ ▄▀▄ █ █ ▄▀  █▄█   ▀█▀ █▄█ ██▀   ▄▀  █▀▄ █ █▀▄ ▄▀▀
#  █▀ █▄▄ ▀▄█ █ █ █▄▄ ▄██    █  ▀▄▀    █  █ █ █▄▄   ▀▄▀▄▀ █▀█ █▄▄ █▄▄ ▄██   █▀█ █ ▀█ █▄▀    █  █ █ █▀▄ ▀▄▀ ▀▄█ ▀▄█ █ █    █  █ █ █▄▄   ▀▄█ █▀▄ █ █▄▀ ▄██
    out_flux = FluxToWallsAndThroughGrids(species, chamber)


#  ▄▀  ▄▀▄ ▄▀▀   █ █▄ █   █ ██▀ ▄▀▀ ▀█▀ █ ▄▀▄ █▄ █
#  ▀▄█ █▀█ ▄██   █ █ ▀█ ▀▄█ █▄▄ ▀▄▄  █  █ ▀▄▀ █ ▀█
    T_injection = 0.03 #à revoir
    gas_injection = GasInjection(species, injection_rates, T_injection, chamber)


#  █ █▄ █ ██▀ █   ▄▀▄ ▄▀▀ ▀█▀ █ ▄▀▀   ▄▀▀ ▄▀▄ █   █   █ ▄▀▀ █ ▄▀▄ █▄ █ ▄▀▀   █   █ █ ▀█▀ █▄█   █ ▄▀▄ █▄ █ ▄▀▀   ▄▀▄ ▄▀▀ ▄▀▀ ██▀ █   ██▀ █▀▄ ▄▀▄ ▀█▀ ██▀ █▀▄   ▀█▀ ▄▀▄   ▀█▀ █▄█ ██▀   █   █ ▄▀▄ █   █   ▄▀▀
#  █ █ ▀█ █▄▄ █▄▄ █▀█ ▄██  █  █ ▀▄▄   ▀▄▄ ▀▄▀ █▄▄ █▄▄ █ ▄██ █ ▀▄▀ █ ▀█ ▄██   ▀▄▀▄▀ █  █  █ █   █ ▀▄▀ █ ▀█ ▄██   █▀█ ▀▄▄ ▀▄▄ █▄▄ █▄▄ █▄▄ █▀▄ █▀█  █  █▄▄ █▄▀    █  ▀▄▀    █  █ █ █▄▄   ▀▄▀▄▀ █▀█ █▄▄ █▄▄ ▄██
    inelastic_collisions = InelasticCollision(species, chamber)

#  █▀▄ █ ▄▀▀ ▄▀▀ ▄▀▄ ▄▀▀ █ ▄▀▄ ▀█▀ █ ▄▀▄ █▄ █
#  █▄▀ █ ▄██ ▄██ ▀▄▀ ▀▄▄ █ █▀█  █  █ ▀▄▀ █ ▀█
    # Delta_E = 0.5 selon pifomètre d'Esteves ( = monoatomic_energy_excess )
    diss1_O2 = Dissociation(species, "O2", "O", get_K_func(species, "O2", "diss1_O2"), 6.12, 0.5, chamber)
    diss2_O2 = Dissociation(species, "O2", "O", get_K_func(species, "O2", "diss2_O2"), 8.40, 0.5, chamber)
    diss_N2 = Dissociation(species, "N2", "N", get_K_func(species, "N2", "diss_N2"), 9.76, 0.5, chamber)

#  ▀█▀ █▄█ ██▀ █▀▄ █▄ ▄█ █ ▄▀▀   █▀▄ █ █▀ █▀ █ █ ▄▀▀ █ ▄▀▄ █▄ █  
#   █  █ █ █▄▄ █▀▄ █ ▀ █ █ ▀▄▄   █▄▀ █ █▀ █▀ ▀▄█ ▄██ █ ▀▄▀ █ ▀█  
    #kappa = lambda T_i : 4.4e-5 * (e / k_B * T_i)**0.8  # noqa: E731
    kappa = dict()
    kappa["N"] = lambda T_i : 1.75*3.95e-4 * (e/k_B * T_i)**0.691 #3.95e-5 * (e/k_B * T_i)**0.691
    kappa["N2"] = lambda T_i : 1.75*2.06e-4 * (e/k_B * T_i)**0.754 #2.06e-5 * (e/k_B * T_i)**0.754
    kappa["O"] =  lambda T_i : 1.75*4.41e-4 * (e/k_B * T_i)**0.679 #4.41e-5 * (e/k_B * T_i)**0.679
    kappa["O2"] = lambda T_i : 1.75*1.66e-4 * (e/k_B * T_i)**0.798 #1.66e-5 * (e/k_B * T_i)**0.798
    #kappa = lambda T_i : 0.0
    th_diff = ThermicDiffusion(species, kappa, 0.03, chamber)

    #soit mettre le kappa en instance, soit faire une liste de kappa (mais faut l'associer à la bonne espèce)
    
    # Reaction list
    reaction_list = [
        exc1_N2, exc2_N2, exc3_N2, exc4_N2, exc5_N2, exc6_N2, exc7_N2, exc8_N2, exc9_N2, exc11_N2, exc12_N2, exc13_N2, exc14_N2, 
        exc1_N, exc2_N, exc1_O2, exc2_O2, exc3_O2, exc4_O2, 
        exc1_O, exc2_O, exc3_O, exc4_O, exc5_O, exc6_O, exc7_O, exc8_O, exc9_O,
        *vib_exc_N2_list, *vib_exc_O2_list, *rot_exc_N2_list, *rot_exc_O2_list,   # * is used to unpack lists (similar to *args in functions)
        ela_N, ela_N2, ela_O, ela_O2, 
        ion_N, ion_O2, ion_N2, ion_O,
        diss1_O2, diss2_O2, diss_N2,
        out_flux, gas_injection, inelastic_collisions, 
        th_diff
    ]

#  ██▀ █   ██▀ ▄▀▀ ▀█▀ █▀▄ ▄▀▄ █▄ █   █▄█ ██▀ ▄▀▄ ▀█▀ █ █▄ █ ▄▀    ██▄ ▀▄▀   ▀█▀ █▄█ ██▀   ▄▀▀ ▄▀▄ █ █    
#  █▄▄ █▄▄ █▄▄ ▀▄▄  █  █▀▄ ▀▄▀ █ ▀█   █ █ █▄▄ █▀█  █  █ █ ▀█ ▀▄█   █▄█  █     █  █ █ █▄▄   ▀▄▄ ▀▄▀ █ █▄▄  
    electron_heating = ElectronHeatingConstantRFPower(species, 1000, chamber)

    return species, initial_state, reaction_list, electron_heating

