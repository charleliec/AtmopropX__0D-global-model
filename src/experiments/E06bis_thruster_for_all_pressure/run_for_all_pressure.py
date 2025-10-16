import sys
import os
import json
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

from reaction_set_N_et_O import get_species_and_reactions


altitude = 250
log_folder_path = Path(__file__).resolve().parent.parent.parent.parent.joinpath("outputs", "logs_for_plot_by_pressure")
# model = GlobalModel(species, reactions_list, chamber, electron_heating, simulation_name="N_O_simple_thruster_constant_kappa", log_folder_path=log_folder_path)

#print(chamber.V_chamber)
# print(chamber.S_eff_total(chamber.n_g_0))
# print(chamber.h_L(chamber.n_g_0))
# print(chamber.h_R(chamber.n_g_0))
# print(chamber.SIGMA_I*chamber.n_g_0)
# print(f"SIGMA I est {chamber.SIGMA_I}")
# print(f"ng0 est {chamber.n_g_0}")


# Solve the model
power_list = [300,600,1000, 1500] #np.arange(1000,1001)
pressure_list = [1e-2, 1.5e-2,2e-2, 3e-2, 5e-2, 7e-2, 1e-1]
final_states = {}
for power in power_list:
    final_states_per_power = {}
    for pressure in pressure_list:
        config_dict = {'R': 6e-2, 'L': 10e-2, 'target_pressure': pressure, 'V_grid': 1000, 'omega': 13.56e6 * 2 * pi, 'N': 5, 'R_coil': 2}
        chamber = Chamber(config_dict)
        species, initial_state, reactions_list, _ = get_species_and_reactions(chamber, altitude)
        electron_heating = ElectronHeatingConstantRFPower(species, power, chamber)
        model = GlobalModel(species, reactions_list, chamber, electron_heating, simulation_name="NO"+str(power)+"_alt_"+str(altitude)+"_pressure_"+str(pressure), log_folder_path=log_folder_path)
        try:
            print("Solving model...")
            sol = model.solve(0, 1, initial_state)  # TODO Needs some testing
            final_states_per_power[pressure] = list(sol.y[:, -1])
            print("Model resolved !")
        except Exception as exception:
            print("Entering exception...")
            model.var_tracker.save_tracked_variables()
            print("Variables saved")
            raise exception
        
with open(os.path.join(log_folder_path, "final_states_for_pressure_and_power.json"), 'w') as file:
            json.dump(final_states, file, indent=4)


for power, pressure_dict in final_states.items():
    pressures = sorted(pressure_dict.keys())
    electron_densities = [pressure_dict[p][0] for p in pressures]
    plt.plot(pressures, electron_densities, marker='o', label=f'{power} W')

plt.xlabel('Pressure')
plt.ylabel('Electron density (m⁻³)')
plt.title('Electron density vs Pressure')
plt.legend(title='Power')
plt.grid(True)
plt.tight_layout()
plt.show()