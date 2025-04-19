import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import k, e, pi
from src.model_components.model import GlobalModel
from src.config import config_dict
from src.model_components.chamber_caracteristics import Chamber
from src.reaction_sets.Reaction_set_Xe_test1 import get_species_and_reactions



chamber = Chamber(config_dict)
species, initial_state, reactions_list, electron_heating = get_species_and_reactions(chamber)
model = GlobalModel(species, reactions_list, chamber, electron_heating, simulation_name="all_reac_current")

#print(chamber.V_chamber)
# print(chamber.S_eff_total(chamber.n_g_0))
# print(chamber.h_L(chamber.n_g_0))
# print(chamber.h_R(chamber.n_g_0))
# print(chamber.SIGMA_I*chamber.n_g_0)
# print(f"SIGMA I est {chamber.SIGMA_I}")
# print(f"ng0 est {chamber.n_g_0}")


# Solve the model
try:
    print("Solving model...")
    #power_list = np.linspace(0,1600,400)
    t0, tf = 0, 1
    #power_list = [50, 100, 200, 400, 600, 800, 1e3, 1.2e3, 1.4e3, 1.6e3]
    coil_currents = [2, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
    #efficiency_list = [0.78, 0.72, 0.68, 0.6, 0.5, 0.48, 0.45, 0.41, 0.37, 0.35]
    #efficiency_list = [0.79, 0.75, 0.67, 0.58, 0.53, 0.46, 0.42, 0.4, 0.37, 0.35]
    final_state = model.solve_for_I_coil(coil_currents, t0, tf, initial_state) # TODO Needs some testing
    print("Model resolved !")
except Exception as exception:
    print("Entering exception...")
    model.var_tracker.save_tracked_variables()
    print("Variables saved")
    raise exception

# Extract time points
#time_points = sol.t

# Create figure and primary axis
# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

# # Plot species concentrations on the first subplot
# for i, specie in enumerate(species.species):
#     L=[]
#     for j in range(len(power_list)):
#         L.append(final_state[j][specie.index])
#     print(L)
#     ax1.plot(power_list, L, label=specie.name)
# ax1.set_ylabel('Density of species (m^-3)')
# ax1.legend(loc='best')
# ax1.grid()

# # Create a secondary y-axis for Xenon temperature
# # ax3 = ax2.twinx()

# # # Plot temperatures: Electron on primary y-axis, Xenon on secondary y-axis
# # ax2.plot(power_list, final_states[species.nb], label='Electron Temp (eV)', color='blue')
# # for i in range(1,3):
# #     ax3.plot(power_list, final_states[species.nb + i], linestyle='--', label= f"Molecules with {i} atoms Temp (eV)")

# # ax2.set_ylabel('Electron Temperature', color='blue')
# # ax3.set_ylabel('Molecules Temperature', color='red')
# # ax2.tick_params(axis='y', labelcolor='blue')
# # ax3.tick_params(axis='y', labelcolor='red')

# # # Combine legends
# # lines_2, labels_2 = ax2.get_legend_handles_labels()
# # lines_3, labels_3 = ax3.get_legend_handles_labels()
# # ax2.legend(lines_2 + lines_3, labels_2 + labels_3, loc='best')

# # Set labels and title
# #ax2.set_xlabel('Time (s)')
# ax1.set_title('Species Concentrations Over Time')
# #ax2.set_title('Temperature Evolution')
# #ax2.grid()

# # Show the plot
# plt.tight_layout()
# plt.show()


# # sol = model.solve(0, 5e-1)    # TODO Needs some testing

# # final_states = sol.y
# # print(final_states)

# # # Plot the results
# # time_points = sol.t
# # fig, ax1 = plt.subplots()

# # # Plot species concentrations on the first y-axis
# # for i, specie in enumerate(species_list.species):
# #     ax1.plot(time_points, final_states[i], label=specie.name)
# # ax1.set_ylabel('Concentration')
# # ax1.tick_params(axis='y')

# # # Create a second y-axis for temperature
# # ax2 = ax1.twinx()
# # for i in range(2):
# #     ax2.plot(time_points, final_states[len(species_list.species) + i], label="Temp " + str(i), linestyle='--')
# # ax2.set_ylabel('Temperature')
# # ax2.tick_params(axis='y', labelcolor='tab:red')

# # # Combine legends from both axes
# # lines_1, labels_1 = ax1.get_legend_handles_labels()
# # lines_2, labels_2 = ax2.get_legend_handles_labels()
# # ax1.legend(lines_1 + lines_2, labels_1 + labels_2, bbox_to_anchor=(0.5, -0.1), ncol=2)

# # plt.xlabel('Time (s)')
# # plt.ylabel('Concentration')
# # plt.title('Species Concentrations Over Time')
# # plt.legend()
# # plt.grid()
# # plt.show()