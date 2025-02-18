import numpy as np
from scipy.constants import pi, e, k, epsilon_0 as eps_0, c, m_e
from scipy.special import jv
from scipy import fsolve

SIGMA_I = 1e-18 # Review this for iodine

def u_B(T_e, m_i):
    return np.sqrt(k * T_e / m_i)

def h_L(n_g, L):
    lambda_i = 1/(n_g * SIGMA_I)
    return 0.86 / np.sqrt(3 + (L / (2 * lambda_i)))

def h_R(n_g, R):
    lambda_i = 1/(n_g * SIGMA_I)
    return 0.8 / np.sqrt(4 + (R / lambda_i))

def maxwellian_flux_speed(T, m):
    # TODO : what is it, where does it come from, does it always hold true
    return np.sqrt((8 * k * T) / (pi * m))

def pressure(T, Q, v, A_out):
    """Calculates pressure in steady state without any plasma.
    T : Temperature in steady state
    Q : inflow rate in the chamber
    v : mean velocity of gas in steady state
    A_out : Effective area for which the gas can leave the chamber"""
    return (4 * k * T * Q) / (v * A_out)

def A_eff(n_g, R, L):
    return (2 * h_R(n_g, R) * pi * R * L) + (2 * h_L(n_g, L) * pi * R**2)

def A_eff_1(n_g, R, L, beta_i):
    return 2 * h_R(n_g, R) * pi * R * L + (2 - beta_i) * h_L(n_g, L) * pi * R**2

def f(v) :
    """répartition maxwellienne des vitesses"""
    a = m_e/(2*pi*k_b*T)
    b = np.exp(-(m_e * v**2)/(k_b*T))
    return 4*pi*v*v*b*(a**(3/2))
cross_section_exc1_N = load_cross_section("exc1_N.csv")
cross_section_exc2_N = load_cross_section("exc2_N.csv")
cross_section_ion_N = load_cross_section("ion_N.csv")
lsite_cross_section_N = [cross_section_exc1_N , cross_section_exc2_N , cross_section_ion_N]

liste_cross_section_N2 = []
liste_cross_section_N2.append(load_cross_section("diss_N2.csv"))
liste_cross_section_N2.append(load_cross_section("exc1_N2.csv"))
liste_cross_section_N2.append(load_cross_section("exc2_N2.csv"))
liste_cross_section_N2.append(load_cross_section("exc3_N2.csv"))
liste_cross_section_N2.append(load_cross_section("exc4_N2.csv"))
liste_cross_section_N2.append(load_cross_section("exc5_N2.csv"))
liste_cross_section_N2.append(load_cross_section("exc6_N2.csv"))
liste_cross_section_N2.append(load_cross_section("exc7_N2.csv"))
liste_cross_section_N2.append(load_cross_section("exc8_N2.csv"))
liste_cross_section_N2.append(load_cross_section("exc9_N2.csv"))
liste_cross_section_N2.append(load_cross_section("exc10_N2.csv"))
liste_cross_section_N2.append(load_cross_section("exc11_N2.csv"))
liste_cross_section_N2.append(load_cross_section("exc12_N2.csv"))
liste_cross_section_N2.append(load_cross_section("exc13_N2.csv"))
liste_cross_section_N2.append(load_cross_section("exc14_N2.csv"))
liste_cross_section_N2.append(load_cross_section("ion_N2.csv"))

liste_cross_section_O2 = []
liste_cross_section_O2.append(load_cross_section("exc1_O2.csv"))
liste_cross_section_O2.append(load_cross_section("exc2_O2.csv"))
liste_cross_section_O2.append(load_cross_section("exc3_O2.csv"))
liste_cross_section_O2.append(load_cross_section("exc4_O2.csv"))
liste_cross_section_O2.append(load_cross_section("diss1_O2.csv"))
liste_cross_section_O2.append(load_cross_section("diss1_O2.csv"))
liste_cross_section_O2.append(load_cross_section("ion_O2.csv"))


def conversion(E) :
    """convertit, pour un électgron, son énergie cinétique en la vitesse correspondante"""
    return min = sqrt(E*1.6*10**(-19) * 2 / m_e)

def nu_i(n_g, liste_cross_section):
    """renvoie pour chaque espèce son nu_m , fréquence de collision en général, calculé par une intégrale approximée"""
    nu_m = 0
    for liste_reaction in liste_cross_section:
        tab_vitesses = []
        for j in range(len(liste_reaction[0])):
            tab_vitesses.append(conversion(liste_reaction[0][j]))
        tab_y = []
        for j in range(len(liste_reaction[1])):
            tab_y.append(n_g * tab_vitesses[j] * f(tab_vitesses[j]) * liste_reaction[1][j])
        nu_m += np.trapz(tab_y,tab_vitesses)
    return nu_m

def eps_i(omega, n_e, n_g, liste_cross_section):
    """calcule la permitivité diélectrique partielle due à une espèce i"""
    omega_pe_sq = (n_e * e**2) / (m_e * eps_0)
    return 1 - (omega_pe_sq / (omega * (omega -  1j*nu_i(n_g, liste_cross_section))))
liste_eps = [eps_i(omega , n_e , n_g , liste_cross_section_O2) , eps_i(omega , n_e , n_g , liste_cross_section_N2) , eps_i(omega , n_e , n_g , liste_cross_section_N)]

###liste des c_i et des eps_i à définir, il faudrait des cross_section sur plus d'espèces

def eps_p (liste_c) :
    """il faut bien définir la liste des c à partir des densités. Les c_i sont normalisés pour que leur somme vaille 1"""
    def equation(x):
        sum = 0
        for i in(range(len(liste_eps))):
            sum += liste_[i](liste_eps[i]-1)/(liste_eps[i] + 2*x)
        return sum + (1-x)/(3*x)
    return fsolve(equation , 1)


def R_ind(R, L, N, omega, n_e, n_g, K_el):
    ep = eps_p(omega, n_e, n_g, K_el)
    k_p = (omega / c) * np.sqrt(ep)
    a = 2 * pi * N**2 / (L * omega * eps_0)
    b = 1j * k_p * R * jv(1, k_p * R) / (ep * jv(0, k_p * R))

    return a * np.real(b)
