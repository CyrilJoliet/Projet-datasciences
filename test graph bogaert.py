# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 19:00:51 2024

@author: cyril
"""

import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

# Lire le fichier CSV avec pandas
df = pd.read_csv('C:/Users/cyril/OneDrive/Documents/cours/M2/DATASCIENCES/AMDG - Sequence STR1-S-2024-10-13-00H06.csv', sep=';')

df['TIME'] = df['TIME'].apply(lambda x: ':'.join([part.zfill(2) for part in x.split(':')]))
print(df.head())



collage_time = df[(df['TIME'] >= "04:14:00") & (df['TIME'] <= '04:15:00')]
colonnes_a_garder = ['DATE', 'TIME', 'SPEED', 'WIDTH', 'LENGTH'] + \
                    [f'M{i}B' for i in range(1, 46)] +[f'F{i}B' for i in range(1, 46)]

collage_fibre= collage_time[colonnes_a_garder]

colonnes_temp = [f'M{i}B' for i in range(1, 46)]+[f'F{i}B' for i in range(1, 46)]

#  variations temperature
for col in colonnes_temp:
    collage_fibre[f'variation_{col}'] = collage_fibre[col].diff()
    

n_mesures = 61# Nombre de mesures (secondes)
n_capteurs = 90

#coordonée X Y
Y = np.arange(n_mesures) * 100  
X = [i * 50 for i in range(45)] + [(i * 50 + 228.5 + 50 * 44) for i in range(45)] 
coord_df = pd.DataFrame()
for i in range(n_capteurs):
    coord_df = pd.concat([coord_df, pd.DataFrame({'X': [X[i]] * n_mesures, 'Y': Y})], ignore_index=True)
    

# joindre donnée et coord X Y 
final_df = pd.DataFrame(columns=['X', 'Y', 'Variation_t'])

for i in range(90):
    if i < 45:
        # Pour les capteurs M1B à M45B
        temp_df = pd.DataFrame({
            'X': [X[i]] * n_mesures,             
            'Y': Y,                              
            'Variation_t': collage_fibre[f'variation_M{i+1}B'],  # Variation pour M
            'MxB': [f'M{i+1}B'] * n_mesures      # Nom du capteur M
        })
    else:
        # Pour les capteurs F1B à F45B
        temp_df = pd.DataFrame({
            'X': [X[i]] * n_mesures,             
            'Y': Y,                              
            'Variation_t': collage_fibre[f'variation_F{i-44}B'],  # Variation pour F
            'MxB': [f'F{i-44}B'] * n_mesures     # Nom du capteur F
        })

    # Ajouter les données du capteur au DataFrame final
    final_df = pd.concat([final_df, temp_df], ignore_index=True)
    
    final_df = final_df[
    ~final_df['MxB'].isin([f'M{i}B' for i in list(range(1, 9)) + list(range(38, 46))]) &  # Retirer M1B-M8B, M38B-M45B
    ~final_df['MxB'].isin([f'F{i}B' for i in list(range(1, 9)) + list(range(38, 46))])    # Retirer F1B-F8B, F38B-F45B
]
    
   
# Décaler les positions X des capteurs FxB de 800 unités
final_df.loc[final_df['MxB'].str.startswith('F'), 'X'] += -800


# Créer une liste des valeurs de X pour lesquelles nous voulons tracer les courbes
X_values = [1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700]

# Initialiser la figure
plt.figure(figsize=(8, 6))

# Boucler sur les valeurs de X et tracer les courbes avec des décalages progressifs pour Y
for i, x in enumerate(X_values):
    df_x = final_df[final_df['X'] == x]
    
    # Appliquer un décalage progressif sur Y
    y_offset = abs(-6 * (i * 50))
    plt.plot(df_x['Y'] + y_offset, df_x['Variation_t'], label=f'X = {x}')

# Titre et labels
plt.xlabel("Y")
plt.ylabel("Variation de Température (°C)")

# Afficher le graphique
plt.show()




df_x_1500 = final_df[final_df['X'] == 1500]

plt.figure(figsize=(8, 6))
plt.scatter(df_x_1500['Y'], df_x_1500['Variation_t'], s=50)
plt.title("Variation de Température pour X=1500 en fonction de Y")
plt.xlabel("Y")
plt.ylabel("Variation de Température (°C)")
plt.show()








plt.scatter(final_df['Y'],final_df["Variation_t"])

variations_t = final_df['Variation_t'].dropna()

Q1 = np.percentile(variations_t, 10)
Q3 = np.percentile(variations_t, 90)
IQR = Q3 - Q1

# Calcul des bornes pour détecter les valeurs hors de l'intervalle IQR
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Détection des valeurs hors de l'intervalle
outliers = variations_t[(variations_t < lower_bound) | (variations_t > upper_bound)]
print(f"Nombre d'outliers détectés: {len(outliers)}")
print(upper_bound,lower_bound)

import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.colors as mcolors
from datetime import timedelta
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")


##34140=14:30->23:59:59    86066

time_ranges = []
start_time = pd.to_datetime("25-06-24 00:01:00")
for i in range(86066):  # 121 secondes de 23:01:00 à 23:03:00
    start_time_str = (start_time + timedelta(seconds=i))
    end_time_str = (start_time + timedelta(seconds=i + 60))
    time_ranges.append((start_time_str, end_time_str))

