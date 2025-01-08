# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 14:41:30 2025

@author: liamg
"""

from scipy.constants import pi, e, k, epsilon_0 as eps_0, c, m_e

 species_list = Species([Specie("e", m_e, -e), Specie("N_2", 4.65e-26, 0), Specie("N", 2.33e-26, 0), Specie("N_2+", 4.65e-26, e), Specie("N+", 2.33e-26, e)])