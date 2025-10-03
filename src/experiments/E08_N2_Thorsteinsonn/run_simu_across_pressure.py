import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import k, e, pi
from pathlib import Path
import pandas as pd

# If global_model_package is not installed as package with pip install -e . , adds the global_model_package to the path so that it can be imported as a package
try :
    import global_model_package
    print("'global_model_package' imported as pip package or already in sys.path.")
except ModuleNotFoundError:
    global_model_package_path = Path(__file__).resolve().parent.parent.parent.joinpath("global_model_package")
    sys.path.append(str(global_model_package_path))

from global_model_package.model import GlobalModel
from global_model_package.chamber_caracteristics import Chamber

from config import config_dict
from reaction_set_N import get_species_and_reactions

log_folder_path = Path(__file__).resolve().parent.parent.parent.parent.joinpath("logs")
outputs_folder_path = Path(__file__).resolve().parent.parent.parent.parent.joinpath("outputs")

pressure_list_mTorr = [0.5, 0.7,1,2,4,8,16,30,50,80,100,110] # in mTor. Multiply by 0.13332237 to get in Pa


pressure_list_Pa = [p*0.13332237 for p in pressure_list_mTorr]

chamber = Chamber(config_dict)
species, initial_state, reactions_list, electron_heating, modifier_func = get_species_and_reactions(chamber)
final_states_list = [] 

for target_pressure in pressure_list_Pa:
    chamber.target_pressure = target_pressure
    print(chamber.target_pressure)
    model = GlobalModel(species, reactions_list, chamber, electron_heating, simulation_name=f"N2_Thorsteinsonn_{target_pressure:.3f}", log_folder_path=log_folder_path)

    # Solve the model
    try:
        print("Solving model...")
        sol = model.solve(0, 1, initial_state, (modifier_func, None))  # TODO Needs some testing
        final_states_list.append(sol.y[:, -1])
        print("Model resolved !")
    except Exception as exception:
        print("Entering exception...")
        model.var_tracker.save_tracked_variables()
        print("Variables saved")
        raise exception


print(final_states_list)
final_states_df = pd.DataFrame(final_states_list, columns=species.names+["T_e", "T_mono", "T_diato"])
final_states_df.to_csv(outputs_folder_path.joinpath("N2_Thorsteinsonn_final_states_across_pressure.csv"))