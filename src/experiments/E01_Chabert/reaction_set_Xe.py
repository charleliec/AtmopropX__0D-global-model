from scipy.constants import pi, e, k, epsilon_0 as eps_0, c, m_e
import numpy as np

from global_model_package.reactions import (Excitation, Ionisation, Dissociation, 
    ThermicDiffusion, InelasticCollision, ElasticCollisionWithElectron, 
    FluxToWallsAndThroughGrids, GasInjection,
    ElectronHeatingConstantRFPower
)

from global_model_package.specie import Species, Specie
from global_model_package.constant_rate_calculation import get_K_func


def get_species_and_reactions(chamber):
                    
    species = Species([Specie("e", m_e, -e, 0, 3/2), Specie("Xe", 2.18e-25, 0, 1, 3/2), Specie("Xe+", 2.18e-25, e, 1, 3/2)])

    init_state = [1.29e18, 2.24e19, 1.29e18, 3.14, 0.03, 0.0001]

    def Kexc(state):
        T=state[species.nb]
        Eexc=11.6
        return 1.2921e-13 * np.exp(-Eexc/T)


    ### Excitation : OK 

    # exc_Xe = Excitation(species, "N", get_K_func(species, "Xe", "exc_Xe"), 11.6, chamber) 
    exc_Xe = Excitation(species, "Xe", get_K_func(species, "Xe", "Ionization_Xe"), 11.6, chamber) 

    #get_K_func(species, "Xe", "exc_Xe")
    #lambda T: Kexc(T)

    ### Ionisation 

    ion_Xe = Ionisation(species, "Xe", "Xe+", get_K_func(species, "Xe", "Ionization_Xe"), 12.127, chamber) 
    # ion_Xe = Ionisation(species, "N", "N+", get_K_func(species, "Xe", "Ionization_Xe"), 12.127, chamber) 
    #get_K_func(species, "Xe", "Ionization_Xe")
    #lambda T: 2.2384710835071163e-15
    #lambda T: Kiz(T)

    ### Elastic Collision : OK
    ela_elec_Xe = ElasticCollisionWithElectron(species, "Xe", get_K_func(species, "Xe", "Elastic_Xe"), chamber) 
    # get_K_func(species, "Xe", "ela_elec_Xe")
    #lambda T : 1e-13

    ### Sortie de Xe à travers les grilles
    out_Xe = FluxToWallsAndThroughGrids(species, chamber) 
    #get_K_func(species, "Xe", "out_Xe")
    
    ### Terme source : OK
    src_Xe = GasInjection(species, [0.0, 1.2e19, 0.0], 0.03, chamber) 

    def Kiz(state):
        T_e=state[species.nb]
        #print(T_e)
        E_iz=12.127
        K_iz_1 = 6.73e-15 * (T_e)**0.5 * (3.97+0.643*T_e - 0.0368 * T_e**2) * np.exp(- E_iz/T_e)
        K_iz_2 = 6.73e-15 * (T_e)**0.5 * (-0.0001031*T_e**2 + 6.386 * np.exp(- E_iz/T_e))
        return 0.5 * (K_iz_1 + K_iz_2)

    ### Inelastic Collision
    in_Xe = InelasticCollision(species, chamber)

    ###Thermic diffusion
    th_Xe = ThermicDiffusion(species, {"Xe" : lambda T_i : 0.0103}, 0.025, chamber)

    reaction_list = [out_Xe, src_Xe, ion_Xe, exc_Xe, ela_elec_Xe, in_Xe, th_Xe] #[exc_Xe, src_Xe] #[exc_Xe, src_Xe, out_Xe] 
    #reaction_list=[ela_elec_Xe]

    electron_heating = ElectronHeatingConstantRFPower(species, 400, chamber) 
    #electron_heating = ElectronHeatingConstantRFPower(species, 1180, chamber)
    #electron_heating = ElectronHeatingConstantCurrent(species, 10, chamber)

    # print([sp.name for sp in species.species if sp.charge == 0])

    return species, init_state, reaction_list, electron_heating
