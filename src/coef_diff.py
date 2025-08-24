import pandas as pd
import io
import os
import numpy as np

# --- Configuration du chemin de votre fichier ---
# Remplacez 'votre_fichier.txt' par le nom exact de votre fichier texte.
# Exemples:
# file_path = 'donnees_N_diffusion.txt'
# file_path = 'donnees_N2_diffusion.txt'
# file_path = 'N_coef-th' # Exemple avec le dernier fichier
# file_path = os.path.join('thermic_coef', 'N_coef-th')
file_path = ".\\thermic_coef\\Xe_coef-th.txt"

try:
    # Lecture du fichier texte
    # sep=r'\s+' : utilise un ou plusieurs espaces/tabulations comme séparateur
    # skip_blank_lines=True : ignore les lignes vides dans le fichier
    df = pd.read_csv(file_path, sep=r'\s+', skip_blank_lines=True)
    print(f"Données chargées avec succès depuis '{file_path}':")

    # --- Nettoyage et renommage des colonnes ---
    # Les noms de colonnes extraits par OCR peuvent être un peu "sales".
    # Cette étape les rend plus faciles à utiliser en Python.

    # 1. Nettoyer les caractères spéciaux et espaces
    # Remplace les caractères non alphanumériques (sauf espaces) et strip les espaces aux extrémités
    df.columns = df.columns.str.replace(r'[^\w\s]', '', regex=True).str.strip()
    # Remplace 'x 106' par 'e6' ou 'E6' pour une meilleure lisibilité
    df.columns = df.columns.str.replace(r'x 106', 'e6', regex=False)
    df.columns = df.columns.str.replace(r'x106', 'e6', regex=False) # Au cas où il n'y aurait pas d'espace

    # 2. Renommer spécifiquement si certains noms sont ambigus ou mal interprétés par l'OCR
    # C'est très important pour les colonnes comme λ', λ'' et λ totale, car l'OCR peut les confondre.
    # Examinez df.columns après l'étape 1 et ajustez ces renommages si nécessaire.
    # Voici une proposition basée sur les tables fournies, l'ordre est important ici.
    # C'est une méthode plus robuste si l'ordre des colonnes est constant.
    # Vérifiez ce que `print(df.columns)` affiche après le chargement initial.

    # Noms de colonnes attendus dans l'ordre de la table:
    # 'T, °K', 'Cp/R', 'η x 10^6', 'λ' x 10^6', 'λ'' x 10^6', 'λ x 10^6'
    # Après le nettoyage général ci-dessus, ils pourraient ressembler à:
    # 'T K', 'Cp/R', 'η e6', 'λ e6', 'λ e6', 'λ e6' (si l'OCR a eu du mal avec les apostrophes)
    # ou 'T K', 'Cp/R', 'η e6', 'λ e6', 'λ_prime_e6', 'λ_double_prime_e6' (si l'OCR a mieux géré)

    # Pour les tables que vous avez fournies, il y a souvent deux colonnes qui contiennent 'λ'
    # et l'une est 'λ'' et l'autre 'λ'.
    # Une approche sûre est de renommer par position si vous êtes sûr de l'ordre:
    # (ATTENTION: Ceci dépend de l'ordre exact et du nombre de colonnes)
    if len(df.columns) == 6: # Vérifiez que vous avez bien 6 colonnes
        # C'est une supposition basée sur les tables fournies. Adaptez si vos en-têtes nettoyés sont différents.
        # Après le nettoyage initial, les noms peuvent être 'T K', 'CpR', 'η e6', 'λ e6', 'λ e6.1', 'λ e6.2'
        # ou des variations. Il faut les inspecter.
        # Tentons une approche plus générique :
        new_column_names = [
            'Temperature_K',
            'Cp_over_R',
            'Eta_e6',        # η (viscosité)
            'Lambda_prime_e6', # λ' (conductivité thermique de translation)
            'Lambda_double_prime_e6', # λ'' (conductivité thermique interne)
            'Lambda_total_e6' # λ (conductivité thermique totale)
        ]
        # Si pandas a déjà fait un renommage automatique pour les doublons, il faut s'adapter
        # Exemple: 'λ e6', 'λ e6.1', 'λ e6.2'
        actual_columns = list(df.columns)
        if len(actual_columns) == len(new_column_names):
             df.columns = new_column_names
        else:
            print("AVERTISSEMENT: Le nombre de colonnes ne correspond pas aux noms prédéfinis. Vérifiez `df.columns`.")
            print("Noms de colonnes après nettoyage initial:", df.columns.tolist())


    print("\nNoms de colonnes après renommage et nettoyage :")
    print(df.columns.tolist())

    print("\nPremières lignes des données :")
    print(df.head())

    print("\nInformations sur les types de données :")
    print(df.info())

    # --- Accéder aux données (exemples) ---
    temp = np.log(df['Temperature_K'])
    cp_over_R = df['Cp_over_R']
    lambda_total = np.log(df['Lambda_total_e6']/(4.185*1e3)) # Ou le nom que vous avez défini pour λ totale

    print(f"\nExemple de la colonne 'Temperature_K' (premières 5 valeurs): {temp.head().tolist()}")
    print(f"Exemple de la colonne 'Lambda_total_e6' (premières 5 valeurs): {lambda_total.head().tolist()}")

    # Les données sont maintenant dans le DataFrame `df` et prêtes pour le traitement.
    # Par exemple, pour l'interpolation:
    # from scipy.interpolate import CubicSpline
    # cs = CubicSpline(df['Temperature_K'], df['Lambda_total_e6'])
    # x_interp = np.linspace(df['Temperature_K'].min(), df['Temperature_K'].max(), 100)
    # y_interp = cs(x_interp)

except FileNotFoundError:
    print(f"Erreur: Le fichier '{file_path}' n'a pas été trouvé.")
    print("Veuillez vous assurer que le fichier est dans le même répertoire que votre script Python, ou spécifiez le chemin complet.")
except Exception as e:
    print(f"Une erreur est survenue lors du chargement ou du traitement du fichier: {e}")


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit

# # --- 1. Générer des données d'exemple (les mêmes que précédemment) ---
# x_data = temp
# y_data = lambda_total

# # --- 2. Définir la fonction de la loi de puissance ---
# def power_law_function(x, a, b):
#     # Important: gérer le cas où x pourrait être non positif pour x**b
#     # Si b est négatif et x=0, cela peut poser problème.
#     # Pour des lois en puissance, x est généralement > 0.
#     return a * (x**b)

# # --- 3. Effectuer l'ajustement avec curve_fit ---
# # p0 est une estimation initiale des paramètres (optionnel mais recommandé pour les non-linéaires)
# # Si les valeurs initiales sont mauvaises, l'optimiseur peut ne pas converger.
# # Ici, je donne des valeurs proches des vraies pour l'exemple.
# # Si vous ne savez pas, essayez (1.0, -1.0) par exemple.
# initial_guess = [2e-5, 0.8]

# # curve_fit retourne les paramètres optimaux et la matrice de covariance
# params, covariance = curve_fit(power_law_function, x_data, y_data, p0=initial_guess)

# # Extraire les paramètres optimaux
# a_fit_nonlinear, b_fit_nonlinear = params

# # Calculer l'erreur standard des paramètres (racine carrée de la diagonale de la matrice de covariance)
# perr = np.sqrt(np.diag(covariance))
# a_std_err, b_std_err = perr

# print(f"\n--- Ajustement par Moindres Carrés (scipy.optimize.curve_fit) ---")
# print(f"Paramètre 'a' ajusté: {a_fit_nonlinear} (Erreur standard: {a_std_err})")
# print(f"Paramètre 'b' ajusté: {b_fit_nonlinear} (Erreur standard: {b_std_err})")

# from scipy.optimize import minimize

# # ... (x_data, y_data, power_law_function définis comme précédemment) ...

# # Fonction de coût (somme des carrés des résidus)
# def cost_function(params, x, y):
#     a, b = params
#     y_pred = power_law_function(x, a, b)
#     return np.sum((y - y_pred)**2)

# initial_guess_minimize = [2e-5, 0.8] # Même type d'estimation initiale

# result = minimize(cost_function, initial_guess_minimize, args=(x_data, y_data), method='Nelder-Mead')

# a_minimize, b_minimize = result.x

# print(f"\n--- Ajustement par Minimisation Générique (Nelder-Mead) ---")
# print(f"Paramètre 'a' ajusté: {a_minimize}")
# print(f"Paramètre 'b' ajusté: {b_minimize:.4f}")

# --- 4. Visualisation des résultats ---
# plt.figure(figsize=(12, 6))

# # Sur une échelle normale
# plt.subplot(1, 2, 1)
# plt.plot(x_data, y_data, 'o', label='Données originales', markersize=4, alpha=0.7)
# #plt.plot(x_data, true_a * x_data**true_b, 'g--', label=f'Vraie loi: $y = {true_a}x^{{{true_b}}}$', linewidth=2)
# plt.plot(x_data, power_law_function(x_data, a_fit_nonlinear, b_fit_nonlinear), 'b-',
#          label=f'Ajustement Non-Linéaire: $y = {a_fit_nonlinear:.2f}x^{{{b_fit_nonlinear:.2f}}}$', linewidth=2)
# plt.title('Ajustement de la loi de puissance (échelle linéaire)')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.grid(True)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# --- 1. Générer des données d'exemple avec une tendance linéaire ---

x_data = temp
# true_m = 2.5
# true_c = 5.0
y_data = lambda_total

# --- 2. Effectuer la régression linéaire ---
slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)

print(f"--- Résultats de la Régression Linéaire (scipy.stats.linregress) ---")
print(f"Pente (m) ajustée: {slope:.4f}")
print(f"Ordonnée à l'origine (c) ajustée: {np.exp(intercept)}")
print(f"Coefficient de corrélation (r): {r_value:.4f}")
print(f"Coefficient de détermination (R²): {r_value**2:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Erreur standard de la pente: {std_err:.4f}")

# --- 3. Générer la ligne ajustée pour la visualisation ---
# y_fitted = slope * x_data + intercept

# # --- 4. Visualisation des résultats ---
# plt.figure(figsize=(8, 6))
# plt.plot(x_data, y_data, 'o', label='Données originales', markersize=5, alpha=0.7)
# plt.plot(x_data, true_m * x_data + true_c, 'g--', label=f'Vraie relation: $y = {true_m}x + {true_c}$', linewidth=2)
# plt.plot(x_data, y_fitted, 'r-', label=f'Régression Linéaire: $y = {slope:.2f}x + {intercept:.2f}$', linewidth=2)
# plt.title('Régression Linéaire Simple')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.grid(True)
# plt.show()
