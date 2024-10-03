# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 19:49:37 2024

@author: cyril
"""

import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.colors as mcolors
from datetime import datetime, timedelta
from matplotlib.animation import FuncAnimation, PillowWriter

# Lire le fichier CSV avec pandas
df = pd.read_csv('C:/Users/cyril/OneDrive/Documents/cours/M2/DATASCIENCES/AMDG - Sequence STR1-S-2024-06-25-14H21.csv', sep=';')

# Formatter l'heure
df['TIME'] = df['TIME'].apply(lambda x: ':'.join([part.zfill(2) for part in x.split(':')]))
print(df.head())

# Fonction pour traiter les données
def process_data(start_time, end_time):
    collage_time = df[(df['TIME'] >= start_time) & (df['TIME'] <= end_time)].copy()
    
    colonnes_a_garder = ['DATE', 'TIME', 'SPEED', 'WIDTH', 'LENGTH'] + \
                        [f'M{i}B' for i in range(1, 46)] + [f'F{i}B' for i in range(1, 46)]

    collage_fibre = collage_time[colonnes_a_garder]

    colonnes_temp = [f'M{i}B' for i in range(1, 46)] + [f'F{i}B' for i in range(1, 46)]

    # Calculer les variations de température
    for col in colonnes_temp:
        collage_fibre.loc[:, f'variation_{col}'] = collage_fibre[col].diff()

    n_mesures = 51  # Nombre de mesures (secondes)
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
start_time = datetime.strptime("23:01:15", "%H:%M:%S")
for i in range(51):  # 121 secondes de 23:01:00 à 23:03:00
    start_time_str = (start_time + timedelta(seconds=i)).strftime("%H:%M:%S")
    end_time_str = (start_time + timedelta(seconds=i + 50)).strftime("%H:%M:%S")
    time_ranges.append((start_time_str, end_time_str))


# Traiter et afficher les données pour chaque plage de temps
#for start_time, end_time in time_ranges:
   # final_df = process_data(start_time, end_time)

    # Création du graphique
 #   norm = mcolors.TwoSlopeNorm(vmin=-1., vmax=2, vcenter=0)
  #  cmap = plt.get_cmap('coolwarm')

 #   plt.figure(figsize=(10, 6))
  #  scatter = plt.scatter(final_df['X'], final_df['Y'], c=final_df['Variation_t'], s=25, cmap=cmap, norm=norm)

 #   cbar = plt.colorbar(scatter)
  #  cbar.set_label('Variation de Température (°C)')
  #  plt.title(f'Variation de Température de {start_time} à {end_time}')
 #   plt.show()

# Créer la figure pour l'animation
fig, ax = plt.subplots(figsize=(10, 6))
norm = mcolors.TwoSlopeNorm(vmin=-1., vmax=2, vcenter=0)
cmap = plt.get_cmap('coolwarm')

# Initialiser le scatter
scatter = ax.scatter([], [], c=[], s=25, cmap=cmap, norm=norm)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Variation de Température (°C)')

# Fonction d'initialisation pour l'animation
def init():
    ax.set_xlim(300, 3500)  # Limites X
    ax.set_ylim(0, 5100)  # Limites Y
    return scatter,

# Fonction de mise à jour pour l'animation
def update(frame):
    start_time, end_time = time_ranges[frame]
    final_df = process_data(start_time, end_time)
    
    scatter.set_offsets(final_df[['X', 'Y']].values)
    scatter.set_array(final_df['Variation_t'].values)
    ax.set_title(f'Variation de Température de {start_time} à {end_time}')
    return scatter,

# Créer l'animation
ani = FuncAnimation(fig, update, frames=len(time_ranges), init_func=init, blit=True)

# Sauvegarder l'animation en tant que GIF
ani.save('temperature_variation.gif', writer=PillowWriter(fps=5))

