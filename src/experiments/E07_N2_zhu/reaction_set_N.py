from scipy.constants import pi, e, k as k_B, epsilon_0 as eps_0, c, m_e
import numpy as np

from global_model_package.reactions import (Excitation, Ionisation, Dissociation, 
                        VibrationalExcitation, RotationalExcitation,
                        ThermicDiffusion, InelasticCollision, ElasticCollisionWithElectron, 
                        PressureBalanceFluxToWalls, OpenWallPressureBalanceFluxToWalls, 
                        GasInjection, ElectronHeatingConstantRFPower
                    )

from global_model_package.specie import Species, Specie
from global_model_package.constant_rate_calculation import get_K_func, ReactionRateConstant

ReactionRateConstant.CROSS_SECTIONS_PATH = "../../../cross_sections"


def get_species_and_reactions(chamber):
    
    species = Species([Specie("e", m_e, -e, 0, 3/2), Specie("N2", 4.65e-26, 0, 2, 5/2), Specie("N", 2.33e-26, 0, 1, 3/2), Specie("N2+", 4.65e-26, e, 2, 5/2), Specie("N+", 2.33e-26, e, 1, 3/2)])
    initial_state_dict = {
        "e": 1.24667316e+16,
        "N2": 1.27095410e+19,
        "N": 9.48200969e+19,
        "N2+": 2.16163824e+15,
        "N+": 1.03050936e+16,
        "T_e": 3.88314080e+00,
        "T_mono": 1.18264462e-01,
        "T_diato": 1.00254543e-01
    }
    compression_rate = 1.
    initial_state = [compression_rate * initial_state_dict[specie.name] for specie in species.species] + [initial_state_dict["T_e"], initial_state_dict["T_mono"], initial_state_dict["T_diato"]]
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

    exc_N2_list = Excitation.from_concatenated_txt_file(species, "N2", "exc_full", "EXCITATION", chamber)

# N
    exc1_N = Excitation(species, "N", get_K_func(species, "N", "exc1_N"), 3.20, chamber)
    exc2_N = Excitation(species, "N", get_K_func(species, "N", "exc2_N"), 4.00, chamber)
    # dion_N2 = Reaction(species_list, "N2", "N, N+", "N+", "e", get_K_func(species_list, "N2", "dion_N2"), 18.00, [1., 1., 1., 1.]) Hassoul 
    exc_N_list = Excitation.from_concatenated_txt_file(species, "N", "exc_IAA_full", "EXCITATION", chamber)

    
#  █ ▄▀▄ █▄ █ █ ▄▀▀ ▄▀▄ ▀█▀ █ ▄▀▄ █▄ █
#  █ ▀▄▀ █ ▀█ █ ▄██ █▀█  █  █ ▀▄▀ █ ▀█
    ion_N = Ionisation(species, "N", "N+", get_K_func(species, "N", "ion_N"), 14.80, chamber)
    ion_N2 = Ionisation(species, "N2", "N2+", get_K_func(species, "N2", "ion_N2"), 15.60, chamber)

#  ██▀ █   ▄▀▄ ▄▀▀ ▀█▀ █ ▄▀▀   ▄▀▀ ▄▀▄ █   █   █ ▄▀▀ █ ▄▀▄ █▄ █ ▄▀▀
#  █▄▄ █▄▄ █▀█ ▄██  █  █ ▀▄▄   ▀▄▄ ▀▄▀ █▄▄ █▄▄ █ ▄██ █ ▀▄▀ █ ▀█ ▄██  # * complete
    ela_N = ElasticCollisionWithElectron(species, "N", get_K_func(species, "N", "ela_N"), chamber)
    ela_N2 = ElasticCollisionWithElectron(species, "N2", get_K_func(species, "N2", "ela_N2"), chamber)

    # ela_N = ElasticCollisionWithElectron(species, "N", lambda T : 1e-13, 0, chamber)
    # ela_N2 = ElasticCollisionWithElectron(species, "N2", lambda T : 1.5e-13, 0, chamber)


#  █ █ █ ██▄ █▀▄ ▄▀▄ ▀█▀ █ ▄▀▄ █▄ █ ▄▀▄ █     ██▀ ▀▄▀ ▄▀▀ █ ▀█▀ ▄▀▄ ▀█▀ █ ▄▀▄ █▄ █
#  ▀▄▀ █ █▄█ █▀▄ █▀█  █  █ ▀▄▀ █ ▀█ █▀█ █▄▄   █▄▄ █ █ ▀▄▄ █  █  █▀█  █  █ ▀▄▀ █ ▀█
    vib_exc_N2_list = VibrationalExcitation.from_concatenated_txt_file(species, "N2", "vib_exc", "EXCITATION", chamber)


#  █▀▄ ▄▀▄ ▀█▀ ▄▀▄ ▀█▀ █ ▄▀▄ █▄ █ ▄▀▄ █     ██▀ ▀▄▀ ▄▀▀ █ ▀█▀ ▄▀▄ ▀█▀ █ ▄▀▄ █▄ █
#  █▀▄ ▀▄▀  █  █▀█  █  █ ▀▄▀ █ ▀█ █▀█ █▄▄   █▄▄ █ █ ▀▄▄ █  █  █▀█  █  █ ▀▄▀ █ ▀█
    rot_exc_N2_list = RotationalExcitation.from_concatenated_txt_file(species, "N2", "rot_exc", "ROTATIONAL", chamber)



#  █▀ █   █ █ ▀▄▀ ██▀ ▄▀▀   ▀█▀ ▄▀▄   ▀█▀ █▄█ ██▀   █   █ ▄▀▄ █   █   ▄▀▀   ▄▀▄ █▄ █ █▀▄   ▀█▀ █▄█ █▀▄ ▄▀▄ █ █ ▄▀  █▄█   ▀█▀ █▄█ ██▀   ▄▀  █▀▄ █ █▀▄ ▄▀▀
#  █▀ █▄▄ ▀▄█ █ █ █▄▄ ▄██    █  ▀▄▀    █  █ █ █▄▄   ▀▄▀▄▀ █▀█ █▄▄ █▄▄ ▄██   █▀█ █ ▀█ █▄▀    █  █ █ █▀▄ ▀▄▀ ▀▄█ ▀▄█ █ █    █  █ █ █▄▄   ▀▄█ █▀▄ █ █▄▀ ▄██
    #out_flux = PressureBalanceFluxToWalls(species, chamber)
    out_flux = OpenWallPressureBalanceFluxToWalls(species, chamber)


#  ▄▀  ▄▀▄ ▄▀▀   █ █▄ █   █ ██▀ ▄▀▀ ▀█▀ █ ▄▀▄ █▄ █
#  ▀▄█ █▀█ ▄██   █ █ ▀█ ▀▄█ █▄▄ ▀▄▄  █  █ ▀▄▀ █ ▀█
    injection_rates = compression_rate * np.array([0.0, 4.4e18, 0.0, 0.0, 0.0]) #, 1e10, 2e13, 1e15, 1e10] #à revoir
    T_injection = 0.03 #à revoir
    gas_injection = GasInjection(species, injection_rates, T_injection, chamber)


#  █ █▄ █ ██▀ █   ▄▀▄ ▄▀▀ ▀█▀ █ ▄▀▀   ▄▀▀ ▄▀▄ █   █   █ ▄▀▀ █ ▄▀▄ █▄ █ ▄▀▀   █   █ █ ▀█▀ █▄█   █ ▄▀▄ █▄ █ ▄▀▀   ▄▀▄ ▄▀▀ ▄▀▀ ██▀ █   ██▀ █▀▄ ▄▀▄ ▀█▀ ██▀ █▀▄   ▀█▀ ▄▀▄   ▀█▀ █▄█ ██▀   █   █ ▄▀▄ █   █   ▄▀▀
#  █ █ ▀█ █▄▄ █▄▄ █▀█ ▄██  █  █ ▀▄▄   ▀▄▄ ▀▄▀ █▄▄ █▄▄ █ ▄██ █ ▀▄▀ █ ▀█ ▄██   ▀▄▀▄▀ █  █  █ █   █ ▀▄▀ █ ▀█ ▄██   █▀█ ▀▄▄ ▀▄▄ █▄▄ █▄▄ █▄▄ █▀▄ █▀█  █  █▄▄ █▄▀    █  ▀▄▀    █  █ █ █▄▄   ▀▄▀▄▀ █▀█ █▄▄ █▄▄ ▄██
    inelastic_collisions = InelasticCollision(species, chamber)

#  █▀▄ █ ▄▀▀ ▄▀▀ ▄▀▄ ▄▀▀ █ ▄▀▄ ▀█▀ █ ▄▀▄ █▄ █
#  █▄▀ █ ▄██ ▄██ ▀▄▀ ▀▄▄ █ █▀█  █  █ ▀▄▀ █ ▀█
    # Delta_E = 0.5 selon pifomètre d'Esteves ( = monoatomic_energy_excess )
    diss_N2 = Dissociation(species, "N2", "N", get_K_func(species, "N2", "diss_N2"), 9.76, 0.5, chamber)

#  ▀█▀ █▄█ ██▀ █▀▄ █▄ ▄█ █ ▄▀▀   █▀▄ █ █▀ █▀ █ █ ▄▀▀ █ ▄▀▄ █▄ █  
#   █  █ █ █▄▄ █▀▄ █ ▀ █ █ ▀▄▄   █▄▀ █ █▀ █▀ ▀▄█ ▄██ █ ▀▄▀ █ ▀█  
    kappa = lambda T_i : 4.4e-5 * (e / k_B * T_i)**0.8  # noqa: E731
    #kappa = lambda T_i : 0.005
    th_diff = ThermicDiffusion(species, kappa, 0.03, chamber)

    #soit mettre le kappa en instance, soit faire une liste de kappa (mais faut l'associer à la bonne espèce)
    
    # Reaction list

    reaction_list = [
        exc1_N2, exc2_N2, exc3_N2, exc4_N2, exc5_N2, exc6_N2, exc7_N2, exc8_N2, exc9_N2, exc11_N2, exc12_N2, exc13_N2, exc14_N2, 
        exc1_N, exc2_N,
        # *exc_N2_list, 
        # *exc_N_list,
        *vib_exc_N2_list, *rot_exc_N2_list,   # * is used to unpack lists (similar to *args in functions)
        ela_N, ela_N2, 
        ion_N, ion_N2, diss_N2,
        out_flux, gas_injection, inelastic_collisions, th_diff
    ]

#  ██▀ █   ██▀ ▄▀▀ ▀█▀ █▀▄ ▄▀▄ █▄ █   █▄█ ██▀ ▄▀▄ ▀█▀ █ █▄ █ ▄▀    ██▄ ▀▄▀   ▀█▀ █▄█ ██▀   ▄▀▀ ▄▀▄ █ █    
#  █▄▄ █▄▄ █▄▄ ▀▄▄  █  █▀▄ ▀▄▀ █ ▀█   █ █ █▄▄ █▀█  █  █ █ ▀█ ▀▄█   █▄█  █     █  █ █ █▄▄   ▀▄▄ ▀▄▀ █ █▄▄  
    electron_heating = ElectronHeatingConstantRFPower(species, 210, chamber)

    def modifier_func_Te(t, state, dy):
        if t<=0.1:
            return dy
        if 0.1 < t:
            dy[5] = (initial_state[5]-state[5])*(t-0.1)/0.1
            return dy
        return dy
    
    def modifier_func_Tmono_diato(t, state, dy):
        if t<=0.1:
            return dy
        if 0.1 < t:
            dy[6] = (10-state[6])*(t-0.1)**2/0.1
            dy[7] = (10-state[7])*(t-0.1)**2/0.1
            print("  dy  :" + " ".join([f"{val:12.5e}" for val in dy]))
            return dy
        return dy
    
    def identity(t, state, dy):
        return dy
    
    def fake_excitation(t, state, dy):
        val = 1e7
        if 0.1>t>0.01:
            print("  dyb :" + " ".join([f"{val:12.5e}" for val in dy]))
            dy[6] += val * (t-0.01)/0.1
            dy[7] += val * (t-0.01)/0.1
            print("  dya :" + " ".join([f"{val:12.5e}" for val in dy]))
        elif t>= 0.1:
            print("  dyb :" + " ".join([f"{val:12.5e}" for val in dy]))
            dy[6] += val 
            dy[7] += val
            print("  dya :" + " ".join([f"{val:12.5e}" for val in dy]))
        return dy

    return species, initial_state, reaction_list, electron_heating, identity

