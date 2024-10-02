# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""
test

import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

# Lire le fichier CSV avec pandas
df = pd.read_csv('C:/Users/cyril/OneDrive/Documents/cours/M2/DATASCIENCES/AMDG - Sequence STR1-S-2024-06-25-14H21.csv',sep=';')

df['TIME'] = df['TIME'].apply(lambda x: ':'.join([part.zfill(2) for part in x.split(':')]))
print(df.head())


collage_time = df[(df['TIME'] >= "22:43:10") & (df['TIME'] <= '22:44:00')]
colonnes_a_garder = ['DATE', 'TIME', 'SPEED', 'WIDTH', 'LENGTH'] + \
                    [f'M{i}B' for i in range(1, 46)] 

collage_fibre= collage_time[colonnes_a_garder]

colonnes_temp = [f'M{i}B' for i in range(1, 46)]

#  variations temperature
for col in colonnes_temp:
    collage_fibre[f'variation_{col}'] = collage_fibre[col].diff()
    

n_mesures = 51  # Nombre de mesures (secondes)
n_capteurs = 45  

#coordonée X Y
Y = np.arange(n_mesures) * 100  
X = [i * 50 for i in range(n_capteurs)]  
coord_df = pd.DataFrame()
for i in range(n_capteurs):
    coord_df = pd.concat([coord_df, pd.DataFrame({'X': [X[i]] * n_mesures, 'Y': Y})], ignore_index=True)
    

# joindre donnée et coord X Y 
final_df = pd.DataFrame(columns=['X', 'Y', 'Variation_t'])

for i in range(n_capteurs):
    temp_df = pd.DataFrame({
        'X': [X[i]] * n_mesures,             
        'Y': Y,                              
        'Variation_t': collage_fibre [f'variation_M{i+1}B'],
        'MxB': [f'M{i+1}B'] * n_mesures
    })

    final_df = pd.concat([final_df, temp_df], ignore_index=True)

#####Graph####
import matplotlib.colors as mcolors
#color
norm = mcolors.Normalize(vmin=final_df['Variation_t'].min(), vmax=final_df['Variation_t'].max())
cmap = plt.get_cmap('coolwarm')  


plt.figure(figsize=(10, 6))
scatter = plt.scatter(final_df['X'], final_df['Y'], c=final_df['Variation_t'], cmap=cmap, norm=norm)


cbar = plt.colorbar(scatter)
cbar.set_label('Variation de Température (°C)')

plt.show()
