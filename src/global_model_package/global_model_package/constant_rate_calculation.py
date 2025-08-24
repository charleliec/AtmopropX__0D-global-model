import os
from pathlib import Path
import re
from typing import TypeVar, Type, Self

import numpy as np 
from numpy.typing import NDArray
import pandas as pd
from scipy.constants import m_e, e, pi, k, epsilon_0 as eps_0, mu_0   # k is k_B -> Boltzmann constant
from scipy.integrate import trapezoid, solve_ivp, odeint
import matplotlib.pyplot as plt
import scienceplots
#from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


#Local modules
from .util import load_csv, load_cross_section
from .specie import Specie, Species
#from specie import Specie, Species

#ReacRateType = TypeVar("ReacRateType", bound="ReactionRateConstant")


class ReactionRateConstant(object):
    """
    ReactionRateConstant : 
    ------
    Class representing the reaction rate constant for a single reaction. __init__ takes as inputs Species, energy_list and cross_section_list but prefer using the utils class methods like from_concatenated_txt_file
    """

    CROSS_SECTIONS_PATH: str|None = None

    def __init__(self, species: Species, energy_list: list[float]|NDArray[np.float64]|pd.Series, cross_section_list: list[float]|NDArray[np.float64]|pd.Series, energy_threshold: float|None = None):
        """
        ReactionRateConstant : 
        ------
        Class representing the reaction rate constant for a single reaction. __init__ takes as inputs Species, energy_list and cross_section_list but prefer using the utils class methods like from_concatenated_txt_file
        """
        self.species: Species = species
        self.energy_treshold: float|None = energy_threshold
        self.energy_list: NDArray[np.float64] = np.array(energy_list)
        self.cross_section_list: NDArray[np.float64] = np.array(cross_section_list)
    
    def __call__(self, state: NDArray[np.float64]) -> float:
        """Calculates the reaction rate constant for a given state, which must be a NDArray[np.float]"""
        #T = T_e * k / e
        T_e = state[self.species.nb]
        # If the table of energy - cross sections doesn't go to high enough temperatures, it is extended by considering that for higher temperatures the cross section stay the same
        if 10 * e * T_e > np.max(self.energy_list):
            self.energy_list = np.append(self.energy_list, np.logspace(np.max(self.energy_list), 10 * e * T_e, 100))
            self.cross_section_list = np.append(self.cross_section_list, [self.cross_section_list[-1]]*100)
        v = np.sqrt(2 * self.energy_list * e / m_e)  # electrons speed
        a = (m_e / (2 * np.pi * e * T_e))**(3/2) * 4 * np.pi
        f = self.cross_section_list * v**3 * np.exp(- m_e * v**2 / (2 * e * T_e)) 
        k_rate: float = trapezoid(a*f, x=v)
        return k_rate

    @classmethod
    def from_concatenated_txt_file(cls, species: Species, file_path: str | Path, reaction_name, has_energy_threshold: bool = False) -> list[Self]:
        """
        Reads the cross-sections concatenated in a file downloaded from lxcat. It return a list of instances of ReactionRateConstant each with the right energy/cross sections lists.

        Parameters
        ----------
        species : Species
        Instance of Species used in simulation
        file_path : str | Path
            Path to the .txt file containing the concatenated cross-sections
        reaction_name: str
            Name of reaction as found on the first line of each reaction block, eg 'EXCITATION', 'IONISATION'...
        has_energy_threshold : bool
            Wether the file contains the energy threshold (on line reaction_name + 8)
        
        Returns
        ----------
        list[ReactionRateConstant]
        """
        assert ReactionRateConstant.CROSS_SECTIONS_PATH is not None, "ReactionRateConstant.CROSS_SECTIONS_PATH must be set before calling from_concatenated_txt_file."
        file_path = os.path.join(ReactionRateConstant.CROSS_SECTIONS_PATH, file_path)
        if has_energy_threshold:
            energy_cs_df_list, energy_threshold_list = cls.parse_concatenated_cross_sections_file(file_path, reaction_name)
            return [cls(species, energy_cs_df["Energy"], energy_cs_df["Cross-section"], energy_threshold) for energy_cs_df, energy_threshold in zip(energy_cs_df_list, energy_threshold_list)] # type: ignore
        else:
            energy_cs_df_list = cls.parse_concatenated_cross_sections_file(file_path, reaction_name)
            return [cls(species, energy_cs_df["Energy"], energy_cs_df["Cross-section"]) for energy_cs_df in energy_cs_df_list] # type: ignore
    
    @staticmethod
    def parse_concatenated_cross_sections_file(file_path: str | Path, reaction_name: str, has_energy_threshold: bool = False) -> tuple[list[pd.DataFrame], list[float]] | list[pd.DataFrame]:
        """
        Reads the cross-sections concatenated in a file downloaded from lxcat. 

        Parameters
        ----------
        file_path : str | Path
            Path to the .txt file containing the concatenated cross-sections
        reaction_name: str
            Name of reaction as found on the first line of each reaction block, eg 'EXCITATION', 'IONISATION'...
        
        Returns
        ----------
        (energy_cross_section_list, energy_threshold_list): tuple[list[pd.DataFrame], list[float]]
            First element of tuple is a list of DataFrame with first column "Energy" and second "Cross-section".
            Second element is the list of energy thresholds.
        """
        print("Starting to parse txt file : ", str(file_path))
        energy_cs_df = []
        energy_thresholds = []

        with open(file_path, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if re.match(r"^\s*\*{5,}", line):
                break
            i +=1

        while i < len(lines):
            line = lines[i].strip()
            # Detect reaction block start
            if reaction_name in line:
                if has_energy_threshold:
                    energy_line = lines[i + 2].strip()
                    try:
                        energy_value = float(re.search(r"(\d+\.\d+(?:e[+-]\d{1,2})?)", energy_line).group(1)) # type: ignore
                    except (ValueError, AttributeError):
                        raise ValueError(f"The energy threshold can't be converted to a float. Here is the line : \"{repr(lines[i+2])}\".")
                i+=3
                # Find start of data table
                while i<len(lines) and not re.match(r"^\s*-{5,}", lines[i]):
                    i+=1
                i+=1
                # assert re.match(r"^\s*-{5,}", lines[i+8]), (f"Text file doesn't seem to have expected format, when looking for -----. You're line looks like : \"{repr(lines[i+8])}\". It should look like the following : \n"
                #                                             "EXCITATION\n"
                #                                             "N2(X,v=0) -> N2(X,v=0)\n"
                #                                             "0.000000e+0\n"
                #                                             "SPECIES: e / N2(X,v=0)\n"
                #                                             "PROCESS: E + N2(X,v=0) -> E + N2(X,v=0), Excitation\n"
                #                                             "PARAM.:  E = 0 eV\n"
                #                                             "UPDATED: 2022-06-16 11:16:45\n"
                #                                             "COLUMNS: Energy (eV) | Cross section (m2)\n"
                #                                             "-----------------------------\n"
                #                                             "1.000000e-2	3.689100e-21\n"
                #                                             "2.000000e-2	3.721100e-21\n"
                #                                             "3.000000e-2	3.753600e-21")
                # i += 9  # skip past dashed line

                # Read the data table
                data = []
                while not re.match(r"^\s*-{5,}", lines[i]):
                    try:
                        parts = lines[i].strip().split()
                        assert len(parts)==2, "More than two parts on one line"
                        data.append([float(part) for part in parts])
                    except Exception as e:
                        raise IOError(f"When reading line \"{repr(lines[i])}\", the following error occured. Check the formatting of the file. \n {type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}") # type: ignore
                    i += 1

                energy_cs_df.append(pd.DataFrame(data, columns=["Energy", "Cross-section"]))
                if has_energy_threshold:
                    energy_thresholds.append(energy_value)

                # # Prepare filename
                # clean_name = re.sub(r"[^\w\-_\. ]", "_", reaction_name.replace("->", "to"))
                # filename = f"{reaction_type}_{clean_name}_E={energy_value:.4f}eV.csv"
                # filepath = os.path.join(output_dir, filename)

                # # Write CSV
                # with open(filepath, "w", newline="") as csvfile:
                #     writer = csv.writer(csvfile)
                #     writer.writerow(["Energy (eV)", "Cross section (m²)"])
                #     for row in data:
                #         writer.writerow(row)

                # block_index += 1

            i += 1  # move to next line
        assert re.match(r"x+",lines[i-1]), f"File not ending with \"xxxx...x\". Ending with : \"{repr(lines[i-1])}\""
        if len(energy_cs_df)==0:
            raise IOError("No cross section table found in this file. Please check its formatting.")
        if has_energy_threshold:
            return energy_cs_df, energy_thresholds
        return energy_cs_df





def rate_constant(T_e, E, cs, m):
    """Calculates a reaction rate constant """
    #T = T_e * k / e
    if 10 * e * T_e > np.max(E):
        E = np.append(E, np.logspace(np.max(E), 10 * e * T_e, 100))
        cs = np.append(cs, [cs[-1]]*100)
    v = np.sqrt(2 * E * e / m)  # electrons speed
    a = (m / (2 * np.pi * e * T_e))**(3/2) * 4 * np.pi
    f = cs * v**3 * np.exp(- m * v**2 / (2 * e * T_e)) 
    k_rate = trapezoid(a*f, x=v)
    return k_rate


def get_K_func(species,specie:str,reaction:str):
    """ specie: string, the specie involved (N, N2 or O2, O)
        reaction: string, the type of reaction (ion_N,exc1_O,...) """
    def get_K(state):
        T_e = state[species.nb]
        cross_sec_path =  Path(__file__).resolve().parent.parent.parent.parent.joinpath(f"cross_sections/{specie}/{reaction}.csv")
        e_r,cs_r=load_cross_section(cross_sec_path)
        k_rate=rate_constant(T_e,e_r,cs_r,m_e) #si on considère que T_e en première approx
        return k_rate 
    return get_K














if __name__ == "__main__":
    # #species_list = Species([Specie("e", 9.1e-31, 0),Specie("Xe0", 10.57e-27, 0), Specie("Xe+", 10.57e-27, 0)])
    species_list = Species([Specie("e", 9.1e-31, -e, 0, 3/2), Specie("Xe", 2.18e-25, 0, 1, 3/2), Specie("Xe+", 2.18e-25, e, 1, 3/2)])
    # state = np.array([10^18,3*10^19,3*10^19,3,0.03])
    # T_e = state[3]
    # #print(state[species_list.nb])

    # get_K_ion=get_K_func(species_list, "Xe", "Ionization_Xe")
    # get_K_exc=get_K_func(species_list, "Xe", "exc_Xe")
    # get_K_el=get_K_func(species_list, "Xe", "Elastic_Xe")

    # T=np.linspace(0.1, 30)
    # k_rate_ion=[get_K_ion(np.array([10^18,3*10^19,3*10^19,t,0.03])) for t in T]
    # k_rate_exc=[get_K_exc(np.array([10^18,3*10^19,3*10^19,t,0.03])) for t in T]
    # k_rate_el=[get_K_el(np.array([10^18,3*10^19,3*10^19,t,0.03])) for t in T]
    # # k_rate_exc=get_K_exc(state)
    # # k_rate_el=get_K_el(state)

    # # print("K_el_exp="+str(k_rate_el))
    # # print("K_ion_exp="+str(k_rate_ion))
    # # print("K_exc_exp="+str(k_rate_exc))

    # #Threshold energies in V
    # E_exc=11.6  #selon les valeurs des données du github: 8.32 eV
    # E_iz=12.13 

    # # #Expressions in Chabert
    # K_el= 1e-13
    # K_exc= 1.2921e-13 * np.exp(- E_exc / T_e)
    # K_iz_1 = 6.73e-15 * (T_e)**0.5 * (3.97+0.643*T_e - 0.0368 * T_e**2) * np.exp(- E_iz/T_e)
    # K_iz_2 = 6.73e-15 * (T_e)**0.5 * (-0.0001031*T_e**2 + 6.386 * np.exp(- E_iz/T_e))
    # K_iz= 0.5 * (K_iz_1 + K_iz_2)

    # def Kiz(T_e):
    #     E_iz=12.127
    #     K_iz_1 = 6.73e-15 * (T_e)**0.5 * (3.97+0.643*T_e - 0.0368 * T_e**2) * np.exp(- E_iz/T_e)
    #     K_iz_2 = 6.73e-15 * (T_e)**0.5 * (-0.0001031*T_e**2 + 6.386 * np.exp(- E_iz/T_e))
    #     return 0.5 * (K_iz_1 + K_iz_2)
    
    # def Kexc(T_e):
    #     Eexc=11.6
    #     return 1.2921e-13 * np.exp(-Eexc/T_e)


    # fig, ax = plt.subplots()
    # ax.plot(T,[1e-13 for t in T], label="Formules de l'article de P. Chabert")
    # ax.plot(T,k_rate_el, label="Avec les sections efficaces")
    # ax.set_xlabel("Température (eV)")
    # ax.set_ylabel(r"$K_{el}$ en $m^{-3}.s^{-1}$")
    # plt.title(r"$K_{el}$ en fonction de la température")

    # # axins = inset_axes(ax, width="30%", height="30%", loc='upper right')
    # # axins.plot(x, y)
    # # axins.plot(x, k_rate_ion)

    # # Définir les limites du zoom
    # # x1, x2, y1, y2 = 5, 12, 0, 0.57e-13
    # # axins.set_xlim(x1, x2)
    # # axins.set_ylim(y1, y2)

    # # mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    # ax.legend()
    # ax.grid()
    # plt.show()

    # # print("K_el="+str(K_el))
    # # print("K_iz="+str(K_iz))
    # # print("K_exc="+str(K_exc))
    # print("starting")
    # constant_rates = ReactionRateConstant.from_concatenated_txt_file(species_list, "D:/Users/Charlelie/Downloads/Cross section.txt", "EXCITATION")
    # print("hello")
    # print(constant_rates[0]([10, 20, 30, 3]))
    # df = df_list[0]
    # print(df.head(10))
    # print(len(df_list))
    # print(len(e_list))
    # print(e_list)
    # print([len(d)+100*e for d, e in zip(df_list, e_list)])
#     species = Species([Specie("e", m_e, -e, 0, 3/2), Specie("N2", 4.65e-26, 0, 2, 5/2), Specie("N", 2.33e-26, 0, 1, 3/2), Specie("N2+", 4.65e-26, e, 2, 5/2), Specie("N+", 2.33e-26, e, 1, 3/2), Specie("O2+", 5.31e-26, e, 2, 5/2), Specie("O2", 5.31e-26, 0, 2, 5/2), Specie("O", 2.67e-26, 0, 1, 3/2), Specie("O+", 2.67e-26, e, 1, 3/2)])

#     initial_state_dict = {
#         "e": 2.1e12,
#         "N2": 8e14,
#         "N": 8e14,
#         "N2+": 1e12,
#         "N+": 1e11,
#         "O2+": 0.5e12,
#         "O2": 2.5e13,
#         "O": 1e15,
#         "O+": 0.5e12,
#         "T_e": 1.0,
#         "T_mono": 0.03,
#         "T_diato": 0.03
#     }
#     compression_rate = 4_000
#     initial_state = [compression_rate * initial_state_dict[specie.name] for specie in species.species] + [initial_state_dict["T_e"], initial_state_dict["T_mono"], initial_state_dict["T_diato"]]

#     T = np.linspace(0.1, 100, 1000)
#     #print(T)

#     import copy

#     def change_T(state, t):
#         # Crée une copie profonde de l'état pour éviter de modifier l'original
#         new_state = copy.deepcopy(state)
#         new_state[species.nb] = t
#         return new_state
    
#     state_list = [change_T(initial_state, t) for t in T]
#     exc_N2_list = np.zeros(len(state_list))
#     exc_N_list = np.zeros(len(state_list))
#     exc_O2_list = np.zeros(len(state_list))
#     exc_O_list = np.zeros(len(state_list))

#     for i in range(1,15):
#         if i != 10:
#             get_K_excN2 = get_K_func(species, "N2", "exc"+str(i)+"_N2")
#             exc_N2_list += np.array([get_K_excN2(state) for state in state_list])
    
#     for i in range(1,3):
#         get_K_excN = get_K_func(species, "N", "exc"+str(i)+"_N")
#         exc_N_list += np.array([get_K_excN(state) for state in state_list])

#     for i in range(1,5):
#         get_K_excO2 = get_K_func(species, "O2", "exc"+str(i)+"_O2")
#         exc_O2_list += np.array([get_K_excO2(state) for state in state_list])

#     for i in range(1,10):
#         get_K_excO = get_K_func(species, "O", "exc"+str(i)+"_O")
#         exc_O_list += np.array([get_K_excO(state) for state in state_list])
        

#     get_K_ionN2 = get_K_func(species, "N2", "ion_N2")
#     ion_N2_list =  np.array([get_K_ionN2(state) for state in state_list])

#     get_K_ionN = get_K_func(species, "N", "ion_N")
#     ion_N_list =  np.array([get_K_ionN(state) for state in state_list])

#     get_K_ionO = get_K_func(species, "O", "ion_O")
#     ion_O_list =  np.array([get_K_ionO(state) for state in state_list])

#     get_K_ionO2 = get_K_func(species, "O2", "ion_O2")
#     ion_O2_list =  np.array([get_K_ionO2(state) for state in state_list])

#     get_K_elaN2 = get_K_func(species, "N2", "ela_N2")
#     ela_N2_list =  np.array([get_K_elaN2(state) for state in state_list])

#     get_K_elaN = get_K_func(species, "N", "ela_N")
#     ela_N_list =  np.array([get_K_elaN(state) for state in state_list])

#     get_K_elaO2 = get_K_func(species, "O2", "ela_O2")
#     ela_O2_list =  np.array([get_K_elaO2(state) for state in state_list])

#     get_K_elaO = get_K_func(species, "O", "ela_O")
#     ela_O_list =  np.array([get_K_elaO(state) for state in state_list])

#     get_K_dissN2 = get_K_func(species, "N2", "diss_N2")
#     diss_N2_list =  np.array([get_K_dissN2(state) for state in state_list])

#     diss_O2_list = np.zeros(len(state_list))
#     for i in range(1,3):
#         get_K_dissO2 = get_K_func(species, "O2", "diss"+str(i)+"_O2")
#         diss_O2_list +=  np.array([get_K_dissO2(state) for state in state_list])

#     get_K_rotN2 = get_K_func(species, "N2", "rot_N2")
#     rot_N2_list =  np.array([get_K_rotN2(state) for state in state_list])

#     get_K_vibN2 = get_K_func(species, "N2", "vib_N2")
#     vib_N2_list =  np.array([get_K_vibN2(state) for state in state_list])

#     get_K_rotO2 = get_K_func(species, "O2", "rot_O2")
#     rot_O2_list =  np.array([get_K_rotO2(state) for state in state_list])

#     get_K_vibO2 = get_K_func(species, "O2", "vib_O2")
#     vib_O2_list =  np.array([get_K_vibO2(state) for state in state_list])

#     #print(state_list)

# #     plt.style.use(['science', 'nature'])
# #     plt.rcParams.update({
# #     "xtick.top": True, "ytick.right": True,
# #     "xtick.direction": "in", "ytick.direction": "in",
# #     "xtick.minor.visible": True, "ytick.minor.visible": True,
# #     "xtick.major.size": 6, "ytick.major.size": 6,
# #     "xtick.minor.size": 3, "ytick.minor.size": 3,
# #     })
# #     fig, ax = plt.subplots(figsize = (6, 5))

# #     ax.plot(T, exc_O2_list, label = "Excitation", linewidth = 2, linestyle = ':')
# #     ax.plot(T, ion_N2_list, label = "Ionization", linewidth = 2, linestyle = '--' ) #(0, (5, 5)
# #     ax.plot(T, ela_N2_list, label = "Elastic", linewidth = 2, linestyle = '-')
# #     # ax.plot(T, diss_N2_list, label = "Dissociation", linewidth = 2, linestyle = '-')
# #     # ax.plot(T, rot_N2_list, label = "Rotation", linewidth = 2, linestyle = '-')
# #     # ax.plot(T, vib_N2_list, label = "Vibration", linewidth = 2, linestyle = '--')
# #     ax.set_xscale('log', base=10)
# #     ax.set_yscale('log', base=10)

# #     leg = plt.legend(frameon=True, fontsize=12, loc = "lower right")
# #     frame = leg.get_frame()
# #     frame.set_facecolor('white')   # Fond blanc
# #     frame.set_edgecolor('black')   # Bordure noire
# #     frame.set_alpha(1)   

# #     # ax.legend(loc='best', fontsize=10, facecolor='white', framealpha=1)

# #     # Définition des limites pour une meilleure visibilité
# #     #ax.set_ylim(0, None)
# #     ax.set_xlim(0.1, 100)
# #     # ax.set_ylim(1e-30, 1e-11)
    
# #     from matplotlib.ticker import LogLocator

# #     # ax.minorticks_on()
# #     # ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=None))
# #     # ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=10))
# #     # ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=None))
# #     # ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=10))


# # # Grands ticks (major) → seulement bas/gauche
# #     ax.tick_params(which='major', direction='in', top=False, right=False, length=6, width=1, labelsize = 9)

# #     # Petits ticks (minor) → seulement haut/droite
# #     ax.tick_params(which='minor', direction='in', top=True, right=True, bottom=True, left=True,
# #                length=3, width=0.8, labelsize = 9)
    

# #     # Supprimer la grille
# #     ax.grid(True)

# #     # Bordure noire sur tout le contour
# #     for spine in ax.spines.values():
# #         spine.set_edgecolor('black')
# #         spine.set_linewidth(1)

# #     # Labels
# #     ax.set_xlabel("Electron Temperature [eV]", fontsize = 12)
# #     ax.set_ylabel(r"Reaction rate [$\mathrm{m^{3}.s^{-1}}$]", fontsize = 12)    

# #     # Export vectoriel haute qualité
# #     plt.tight_layout()
# #     #plt.title(r"Reaction rate for $N_2$ as a function of electron temperature", fontsize = 12)
# #     plt.savefig("reaction_rate_N.pdf", dpi=300)
# #     plt.show()
#     import matplotlib.pyplot as plt
#     import numpy as np
#     from matplotlib.ticker import LogLocator, NullFormatter, LogFormatter

#     # --- Style publication ---
#     plt.style.use(['science', 'nature', 'no-latex'])
#     plt.rcParams.update({
#         "xtick.direction": "in",
#         "ytick.direction": "in",
#         "xtick.top": True,
#         "ytick.right": True,
#         "xtick.minor.visible": True,
#         "ytick.minor.visible": True,
#         "xtick.major.size": 6,
#         "ytick.major.size": 6,
#         "xtick.minor.size": 3,
#         "ytick.minor.size": 3,
#         "xtick.labelsize": 8,
#         "ytick.labelsize": 8,
#         "axes.labelpad": 15
#     })

#     # Données X
#     x = np.logspace(0.1, 2, 300)

#     # --- Courbes spécifiques à chaque subplot ---
#     subplot_data = [
#         # Subplot A
#         [
#             (T, exc_N2_list, 'Excitation', ':', 'C0'),
#             (T, ion_N2_list, 'Ionization', '--', 'C1'),
#             (T, ela_N2_list, 'Elastic', '-', 'C2'),
#             (T, diss_N2_list, 'Dissociation', ':', 'C3'),
#             (T, rot_N2_list, 'Rotation', '-', 'C4'),
#             (T, vib_N2_list, 'Vibration', '--', 'C5')
#         ],
#         # Subplot B
#         [
#             (T, exc_N_list, 'Excitation', ':', 'C0'),
#             (T, ion_N_list, 'Ionization', '--', 'C1'),
#             (T, ela_N_list, 'Elastic', '-', 'C2')
#         ],
#         # Subplot C
#         [
#             (T, exc_O2_list, 'Excitation', ':', 'C0'),
#             (T, ion_O2_list, 'Ionization', '--', 'C1'),
#             (T, ela_O2_list, 'Elastic', '-', 'C2'),
#             (T, diss_O2_list, 'Dissociation', ':', 'C3'),
#             (T, rot_O2_list, 'Rotation', '-', 'C4'),
#             (T, vib_O2_list, 'Vibration', '--', 'C5')
#         ],
#         # Subplot D
#         [
#             (T, exc_O_list, 'Excitation', ':', 'C0'),
#             (T, ion_O_list, 'Ionization', '--', 'C1'),
#             (T, ela_O_list, 'Elastic', '-', 'C2'),
#         ]
#     ]

#     subplot_titles = [r"a) $\mathrm{N_2}$", r"b) N", r"c) $\mathrm{O_2}$", r"d) O"]
#     subplot_letters = ["A", "B", "C", "D"]

#     # --- Création figure ---
#     fig, axs = plt.subplots(2, 2, figsize=(8, 6))

#     for ax, curves, title, letter in zip(axs.flat, subplot_data, subplot_titles, subplot_letters):
#         # Ajout des courbes propres à ce subplot
#         for X, Y, label, style, color in curves:
#             ax.plot(X, Y, linestyle=style, color=color, lw=1.2, label=label)
        
#         # Échelle log
#         ax.set_xscale('log')
#         ax.set_yscale('log')
        
#         ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=10))
#         ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2,10)*0.1, numticks=10))
#         ax.yaxis.set_major_locator(LogLocator(base=10.0))
#         ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2,10)*0.1, numticks=10))

#         # ax.xaxis.set_major_formatter(LogFormatter(base=10.0))

        
#         # Ticks majeurs bas/gauche seulement
#         # ax.tick_params(which='major', top=False, right=False, length=6, width=1)
#         # # Mineurs sur tout le contour
#         # ax.tick_params(which='minor', top=True, right=True, bottom=True, left=True,
#         #             length=1, width=0.3)
        
#         # Bordures noires
#         for s in ax.spines.values():
#             s.set_linewidth(1)
        
#         # Lettre en haut à gauche
#         # ax.text(0.05, 0.95, f"({letter})", transform=ax.transAxes,
#         #         fontsize=9, va='top', ha='left')

#         ax.set_xlim(0.1, 100)
#         ax.set_ylim(1e-30, 1e-11)
        
#         # Titre du subplot
#         ax.set_title(title, fontsize=9)
#         ax.legend(frameon=True, facecolor='white', framealpha=1, fontsize=7, loc = "lower right")

#         ax.grid(True)

#     # for ax, curves, title, letter in zip(axs.flat, subplot_data, subplot_titles, subplot_letters):
#     #     ax.xaxis.set_major_formatter(LogFormatter(base=10))
#     #     ax.xaxis.set_minor_formatter(NullFormatter())
#     #     ax.yaxis.set_major_formatter(LogFormatter(base=10))
#     #     ax.yaxis.set_minor_formatter(NullFormatter())

#         # Légendes axes
#         for ax in axs[1, :]:
#             ax.set_xlabel("Electron Temperature [eV]", fontsize = 12)   
#         for ax in axs[:, 0]:
#             ax.set_ylabel(r"Reaction rate [$\mathrm{m^{3}.s^{-1}}$]", fontsize = 12) 
    
#     # fig.supxlabel("Electron Temperature [eV]", fontsize=12)
#     # fig.supylabel(r"Reaction rate [$\mathrm{m^{3}.s^{-1}}$]", fontsize=12)
    
#     # Légende propre à ce subplot
    
        

#     # Titre global
#     fig.suptitle("Reaction rates for considered species as a function of electron temperature", fontsize=14)

#     plt.tight_layout(rect=[0, 0.15, 1, 0.95])
#     fig.subplots_adjust(bottom=0.15)
#     plt.savefig("reaction_rate_all.pdf", dpi=300)
#     plt.show()




    
