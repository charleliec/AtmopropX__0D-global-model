import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import k, e, pi
from pathlib import Path

# If global_model_package is not installed as package with pip install -e . , adds the global_model_package to the path so that it can be imported as a package
try :
    import global_model_package
    print("'global_model_package' imported as pip package or already in sys.path.")
except ModuleNotFoundError:
    global_model_package_path = Path(__file__).resolve().parent.parent.parent.joinpath("global_model_package")
    sys.path.append(str(global_model_package_path))

from global_model_package.model import GlobalModel
from global_model_package.chamber_caracteristics import Chamber
from global_model_package.reactions import ElectronHeatingConstantRFPower

from config import config_dict
from reaction_set_N import get_species_and_reactions

power_list = [110, 160, 210, 260, 310, 410, 500]
e_density_list = []
for power in power_list:
    chamber = Chamber(config_dict)
    species, initial_state, reactions_list, _, modifier_func = get_species_and_reactions(chamber)
    log_folder_path = Path(__file__).resolve().parent.parent.parent.parent.joinpath("logs")

    electron_heating = ElectronHeatingConstantRFPower(species, power, chamber)
    model = GlobalModel(species, reactions_list, chamber, electron_heating, simulation_name="N2_Zhu", log_folder_path=log_folder_path)

    # Solve the model
    try:
        print("Solving model...")
        sol = model.solve(0, 1, initial_state)  # TODO Needs some testing
        print("Model resolved !")
    except Exception as exception:
        print("Entering exception...")
        model.var_tracker.save_tracked_variables()
        print("Variables saved")
        raise exception
    final_state = sol.y[:, -1]
    e_density_list.append(final_state[0])

real_data = np.array([2734871572.052651,6468607661.546294, 9108764197.647696, 12052609368.708414])*1e6
real_pow = [110,210,310,410]

plt.plot(power_list, e_density_list, label="model", marker="o")
plt.plot(real_pow,real_data, label="real", marker="^", linestyle="--")
plt.legend()
plt.show()
