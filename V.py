# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 19:49:37 2024

@author: cyril
"""

import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.colors as mcolors
from datetime import timedelta
from sklearn.linear_model import LinearRegression

# Lire le fichier CSV avec pandas
df = pd.read_csv('C:/Users/vanas/Documents/M2/LBIRE2234/AMDG - Sequence STR1-S-2024-06-25-14H21.csv', sep=';')
start = pd.to_datetime("25-06-24 14:21:18")  # Temps de départ

df['TIME'] = [(start + timedelta(seconds=i)) for i in range(len(df))]


# Fonction pour traiter les données
def process_data(start_time, end_time):
    collage_time = df[(df['TIME'] >= start_time) & (df['TIME'] <= end_time)]
    
    colonnes_a_garder = ['DATE', 'TIME', 'SPEED', 'WIDTH', 'LENGTH'] + \
                        [f'M{i}B' for i in range(1, 46)] + [f'F{i}B' for i in range(1, 46)]

    collage_fibre = collage_time[colonnes_a_garder]

    colonnes_temp = [f'M{i}B' for i in range(1, 46)] + [f'F{i}B' for i in range(1, 46)]

    # Calculer les variations de température
    for col in colonnes_temp:
        collage_fibre.loc[:, f'variation_{col}'] = collage_fibre[col].diff()

    n_mesures = 61  # Nombre de mesures (secondes)
    n_capteurs = 90

    # Coordonnées X Y
    Y = np.arange(n_mesures) * 100  
    X = [i * 50 for i in range(45)] + [(i * 50 + 228.5 + 50 * 44) for i in range(45)] 
    final_df = pd.DataFrame(columns=['X', 'Y', 'Variation_t'])

    for i in range(n_capteurs):
        if i < 45:
            temp_df = pd.DataFrame({
                'X': [X[i]] * n_mesures,             
                'Y': Y,                              
                'Variation_t': collage_fibre[f'variation_M{i+1}B'],  # Variation pour M
                'MxB': [f'M{i+1}B'] * n_mesures      # Nom du capteur M
            })
        else:
            temp_df = pd.DataFrame({
                'X': [X[i]] * n_mesures,             
                'Y': Y,                              
                'Variation_t': collage_fibre[f'variation_F{i-44}B'],  # Variation pour F
                'MxB': [f'F{i-44}B'] * n_mesures     # Nom du capteur F
            })

        final_df = pd.concat([final_df, temp_df], ignore_index=True)

    # Filtrer les capteurs
    final_df = final_df[
        ~final_df['MxB'].isin([f'M{i}B' for i in list(range(1, 9)) + list(range(38, 46))]) &  # Retirer M1B-M8B, M38B-M45B
        ~final_df['MxB'].isin([f'F{i}B' for i in list(range(1, 9)) + list(range(38, 46))])    # Retirer F1B-F8B, F38B-F45B
    ]

    # Décaler les positions X des capteurs FxB de 800 unités
    final_df.loc[final_df['MxB'].str.startswith('F'), 'X'] += -800

    return final_df

# Générer les plages de temps de 23:01:00 à 23:03:00 avec un décalage de 1 seconde
time_ranges = []
start_time = pd.to_datetime("25-06-24 22:42:10")
for i in range(61):  # 121 secondes de 23:01:00 à 23:03:00
    start_time_str = (start_time + timedelta(seconds=i))
    end_time_str = (start_time + timedelta(seconds=i + 60))
    time_ranges.append((start_time_str, end_time_str))


for start_time, end_time in time_ranges:
    final_df = process_data(start_time, end_time)
    
    seuil = 1
    filtered_df = final_df[final_df['Variation_t'] > seuil]

    # Vérifier s'il y a des points filtrés avant de continuer
    if filtered_df.empty:
        
        norm = mcolors.TwoSlopeNorm(vmin=-1., vmax=2,vcenter=0)
        cmap = plt.get_cmap('coolwarm')  


        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(final_df['X'], final_df['Y'], c=final_df['Variation_t'],s=25, cmap=cmap, norm=norm)
        plt.xlim([300, 3600])  # Remplace 'min_x' et 'max_x' par les limites souhaitées pour l'axe X
        plt.ylim([-100, 6300])

        cbar = plt.colorbar(scatter)
        cbar.set_label('Variation de Température (°C)')
        plt.title(f'Variation de Température de {start_time.time()} à {end_time}')

        plt.show()
    else: 
    
    # 2. Trouver le point de la base du "V" (celui avec le plus petit Y)
        min_Y_point = filtered_df.loc[filtered_df['Y'].idxmin()]

    # Extraire ses coordonnées X et Y
        base_X = min_Y_point['X']
        base_Y = min_Y_point['Y']

    # 3. Séparer les données en deux groupes : à gauche et à droite de ce point
        left_df = filtered_df[filtered_df['X'] <= base_X]
        right_df = filtered_df[filtered_df['X'] >= base_X]

    # 4. Effectuer une régression linéaire pour chaque groupe
    # Régression pour la partie gauche
        X_left = left_df['X'].values.reshape(-1, 1)
        y_left = left_df['Y'].values
        model_left = LinearRegression()
        model_left.fit(X_left, y_left)
        y_left_pred = model_left.predict(X_left)

    # Régression pour la partie droite
        X_right = right_df['X'].values.reshape(-1, 1)
        y_right = right_df['Y'].values
        model_right = LinearRegression()
        model_right.fit(X_right, y_right)
        y_right_pred = model_right.predict(X_right)

    # Création du graphique
        norm = mcolors.TwoSlopeNorm(vmin=-1., vmax=2, vcenter=0)
        cmap = plt.get_cmap('coolwarm')

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(final_df['X'], final_df['Y'], c=final_df['Variation_t'], s=25, cmap=cmap, norm=norm)

        plt.plot(X_left, y_left_pred, color='black', label=f'Régression gauche\nPente: {model_left.coef_[0]:.2f}')
        # Tracer la droite de régression pour la partie droite
        plt.plot(X_right, y_right_pred, color='black', label=f'Régression droite\nPente: {model_right.coef_[0]:.2f}')
        
        plt.xlim([300, 3550])  # Remplace 'min_x' et 'max_x' par les limites souhaitées pour l'axe X
        plt.ylim([-100, 6200])

        cbar = plt.colorbar(scatter)
        cbar.set_label('Variation de Température (°C)')
        plt.title(f'Variation de Température de {start_time.time()} à {end_time.time()}')
        plt.show()


#------------------------------ Avec simulateur tableau de bord

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.linear_model import LinearRegression

# Fonction pour calculer l'angle entre deux pentes
def calculate_angle(m1, m2):
    angle_radians = np.arctan(np.abs((m2 - m1) / (1 + m1 * m2)))
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees

# Fonction pour vérifier si deux pentes sont de signes opposés
def have_opposite_slopes(m1, m2):
    return m1 * m2 < 0

# Boucle sur les plages horaires
for start_time, end_time in time_ranges: 
    final_df = process_data(start_time, end_time)
    
    seuil = 1
    filtered_df = final_df[final_df['Variation_t'] > seuil]

    # Vérifier s'il y a des points filtrés avant de continuer
    if filtered_df.empty:
        norm = mcolors.TwoSlopeNorm(vmin=-1., vmax=2, vcenter=0)
        cmap = plt.get_cmap('coolwarm')  

        fig, ax = plt.subplots(figsize=(10, 6)) #plt.figure(figsize=(10, 6))
        scatter = plt.scatter(final_df['X'], final_df['Y'], c=final_df['Variation_t'], s=25, cmap=cmap, norm=norm)
        plt.xlim([300, 3600])  
        plt.ylim([-100, 6300])
        
        #Déterminer la couleur de fond en fonction des conditions
        background_color = 'green'  # Par défaut
        fig.patch.set_facecolor(background_color)  # Changer la couleur de fond ici
        norm = mcolors.TwoSlopeNorm(vmin=-1., vmax=2, vcenter=0)
        cmap = plt.get_cmap('coolwarm')

        cbar = plt.colorbar(scatter)
        cbar.set_label('Variation de Température (°C)')
        plt.title(f'Variation de Température de {start_time.time()} à {end_time}')

        plt.show()
    else: 
        # 2. Trouver le point de la base du "V" (celui avec le plus petit Y)
        min_Y_point = filtered_df.loc[filtered_df['Y'].idxmin()]

        # Extraire ses coordonnées X et Y
        base_X = min_Y_point['X']
        base_Y = min_Y_point['Y']

        # 3. Séparer les données en deux groupes : à gauche et à droite de ce point
        left_df = filtered_df[filtered_df['X'] <= base_X]
        right_df = filtered_df[filtered_df['X'] >= base_X]

        # 4. Effectuer une régression linéaire pour chaque groupe
        # Régression pour la partie gauche
        X_left = left_df['X'].values.reshape(-1, 1)
        y_left = left_df['Y'].values
        model_left = LinearRegression()
        model_left.fit(X_left, y_left)
        y_left_pred = model_left.predict(X_left)
        slope_left = model_left.coef_[0]

        # Régression pour la partie droite
        X_right = right_df['X'].values.reshape(-1, 1)
        y_right = right_df['Y'].values
        model_right = LinearRegression()
        model_right.fit(X_right, y_right)
        y_right_pred = model_right.predict(X_right)
        slope_right = model_right.coef_[0]

        # 5. Déterminer la couleur de fond en fonction des conditions
        #background_color = 'green'  # Par défaut

        # Vérification des pentes opposées et de l'angle
        if have_opposite_slopes(slope_left, slope_right):
            angle = calculate_angle(slope_left, slope_right)
            if 15 <= angle <= 120:
                background_color = 'orange'

        # Vérification des pentes constantes pendant un intervalle de temps de 10s avec une marge d'erreur
        tolerance = 2  # Définit la tolérance pour considérer les pentes comme "constantes" (par exemple, 5%)
        time_interval = (end_time - start_time).total_seconds()

        # Si l'intervalle de temps est supérieur ou égal à 10 secondes, on vérifie si les pentes sont constantes
        if time_interval >= 10 and abs(slope_left - slope_right) <= tolerance * abs(slope_left):
            background_color = 'red'

        # 6. Création du graphique avec changement de couleur de fond
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor(background_color)  # Changer la couleur de fond ici

        norm = mcolors.TwoSlopeNorm(vmin=-1., vmax=2, vcenter=0)
        cmap = plt.get_cmap('coolwarm')

        scatter = ax.scatter(final_df['X'], final_df['Y'], c=final_df['Variation_t'], s=25, cmap=cmap, norm=norm)

        # Tracer la droite de régression pour la partie gauche
        ax.plot(X_left, y_left_pred, color='black', label=f'Régression gauche\nPente: {slope_left:.2f}')
        # Tracer la droite de régression pour la partie droite
        ax.plot(X_right, y_right_pred, color='black', label=f'Régression droite\nPente: {slope_right:.2f}')
        
        plt.xlim([300, 3550])  
        plt.ylim([-100, 6200])

        cbar = plt.colorbar(scatter)
        cbar.set_label('Variation de Température (°C)')
        plt.title(f'Variation de Température de {start_time.time()} à {end_time.time()}')

        plt.legend()
        plt.show()

        
