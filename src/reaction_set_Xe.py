from scipy.constants import pi, e, k, epsilon_0 as eps_0, c, m_e
from src.reactions.excitation_reaction import Excitation
from src.reactions.ionisation_reaction import Ionisation
from src.reactions.elastic_collision_with_electrons_reaction import ElasticCollisionWithElectron
from src.reactions.flux_to_walls_and_grids_reaction import FluxToWallsAndThroughGrids
from src.reactions.gas_injection_reaction import GasInjection
from src.specie import Species, Specie
from src.constant_rate_calculation import get_K_func

def get_species_and_reactions(chamber):
  species = Species([Specie("e", m_e, -e, 0, 0), Specie("Xe", 2.18e-25, 0, 1, 3/2), Specie("Xe+", 2.18e-25, e1, 3/2)])
  
  ### Ionisation
  ion_Xe = Ionisation(species, "Xe", "Xe+", get_K_func(species, "Xe", "ion_Xe"), 12.127, chamber) 
  
  ### Excitation
  exc_Xe = Excitation(species, "Xe", get_K_func(species, "Xe", "exc_Xe"), 11.6, chamber) 
  
  ### Terme source
  src_Xe = GasInjection(species, [0.0, 1.2e-19, 0.0], 500, chamber) 
  
  ### Recombinaison ionique
  rec_Xe = FluxToWallsAndThroughGrids(species, "Xe+", get_K_func(species, "Xe+", "rec_Xe"), 0, chamber) 
  
  ### Sortie de Xe à travers les grilles
  out_Xe = FluxToWallsAndThroughGrids(species, "Xe", get_K_func(species, "Xe", "out_Xe"), 0, chamber) 
  
  ### Sortie de Xe+ à travers les grilles
  out_Xe_plus = FluxToWallsAndThroughGrids(species, "Xe+", get_K_func(species, "Xe+", "out_Xe+"), 0, chamber) 
  
  ### Choc élastique électron-neutre
  ela_elec_Xe = ElasticCollisionWithElectron(species, "Xe", get_K_func(species, "Xe", "ela_elec_Xe"), 0, chamber) 
  
  ### Choc élastique ion-neutre
  # On néglige pour le moment 
  
  ### Diffusion thermique vers les parois  --> on avait trouvé par un savant calcul d'ODG que c'est négligeable
  
  # Reaction list
  reaction_list = [ion_Xe, exc_Xe, src_Xe, rec_Xe, out_Xe, out_Xe_plus, ela_elec_Xe]

  return species, reaction_list
