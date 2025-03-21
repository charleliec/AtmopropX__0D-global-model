import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import k, e, pi
from src.model import GlobalModel
from src.config import config_dict
from src.chamber_caracteristics import Chamber


print("start")

if __name__ == "__main__":
    chamber = Chamber(config_dict)
    model = GlobalModel()

    # Solve for several values of I_coil

    I_coil = np.linspace(1, 40, 20)
    p, s = model.solve_for_I_coil(I_coil)

    T_e = s[:, 0]
    T_g = s[:, 1]
    n_e = s[:, 2]
    n_g = s[:, 3]

    print("plot start")

    thrust = model.eval_property(model.thrust_i, s)
    j_i = model.eval_property(model.j_i, s)
    plt.ylim(0, 200)
    plt.xlim(0, 1600)
    plt.plot(p, j_i)
    plt.title("Current density as function of power in the coil")
    plt.show(block=False)
    print("thruster plot")

    #exit()
    # Temperature plot

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax2.plot(p, T_e * k / e, 'g-')
    ax1.plot(p, T_g, 'b-')

    ax1.set_xlabel('$P_{RF}$ [W]')
    ax2.set_ylabel('$T_e$ [eV]', color='g')
    ax1.set_ylabel('$T_g$ [K]', color='b')

    ax1.set_xlim((0, 1600))
    ax2.set_ylim(((T_e * k / e).min(), 5.3))
    ax1.set_ylim((255, 540))

    plt.title("Gas and electron temperature as function of power")

    plt.show(block=False)
    print("2nd plot")

    # Density plot
    plt.xlim((0, 1600))
    plt.semilogy(p, n_e, label='$n_e$')
    plt.semilogy(p, n_g, label='$n_g$')
    plt.xlabel('n [$m^{-3}$]')
    plt.xlabel('$P_{RF}$ [W]')
    plt.legend()
    plt.title("Gas and electron (aka ion) density as function of power")
    plt.show()
    print("desity plot")

