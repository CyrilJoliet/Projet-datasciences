# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 17:55:49 2024

@author: cyril
"


# -*- coding: utf-8 -*-
"""

import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.colors as mcolors
from datetime import timedelta
from sklearn.linear_model import LinearRegression
import warnings
import os
from PIL import Image
warnings.filterwarnings("ignore")


# Lire le fichier CSV avec pandas
df= pd.read_csv(r"C:\Users\cyril\OneDrive\Documents\cours\M2\DATASCIENCES\AMDG - Sequence STR1-S-2024-06-25-14H21 OK 2col.csv", sep=';')
df['DATETIME'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'],format='%y/%m/%d %H:%M:%S')

# Accéder au premier élément de la colonne 'DATETIME'
start = df['DATETIME'][0]
df['TIME'] = [(start + timedelta(seconds=i)) for i in range(len(df))]


# Fonction pour traiter les données
def process_data(start_time, end_time, fibre='B'):
    # Vérification de la validité de l'argument fibre
    if fibre not in ['B', 'D']:
        raise ValueError("Le paramètre 'fibre' doit être 'B' ou 'D'")
    
    # Filtrer les données sur la période souhaitée
    collage_time = df[(df['TIME'] >= start_time) & (df['TIME'] <= end_time)]
    
    # Colonnes à garder, adaptant MxB/FxB ou MxD/FxD selon fibre
    colonnes_a_garder = ['DATE', 'TIME', 'SPEED', 'WIDTH', 'LENGTH'] + \
                        [f'M{i}{fibre}' for i in range(1, 46)] + [f'F{i}{fibre}' for i in range(1, 46)]

    collage_fibre = collage_time[colonnes_a_garder]

    # Colonnes pour les variations de température (BxB ou DxD selon fibre)
    colonnes_temp = [f'M{i}{fibre}' for i in range(1, 46)] + [f'F{i}{fibre}' for i in range(1, 46)]

    # Calculer les variations de température
    for col in colonnes_temp:
        collage_fibre[f'variation_{col}'] = collage_fibre[col].diff()

    n_mesures = 61  # Nombre de mesures (secondes)
    n_capteurs = 90

    # Coordonnées X et Y
    Y = np.arange(n_mesures) * 100  
    X = [i * 50 for i in range(45)] + [(i * 50 + 228.5 + 50 * 44) for i in range(45)] 
    final_df = pd.DataFrame(columns=['X', 'Y', 'Variation_t'])

    for i in range(n_capteurs):
        if i < 45:
            temp_df = pd.DataFrame({
                'X': [X[i]] * n_mesures,             
                'Y': Y,                              
                'Variation_t': collage_fibre[f'variation_M{i+1}{fibre}'],  # Variation pour M
                'MxB': [f'M{i+1}{fibre}'] * n_mesures  # Nom du capteur M
            })
        else:
            temp_df = pd.DataFrame({
                'X': [X[i]] * n_mesures,             
                'Y': Y,                              
                'Variation_t': collage_fibre[f'variation_F{i-44}{fibre}'],  # Variation pour F
                'MxB': [f'F{i-44}{fibre}'] * n_mesures  # Nom du capteur F
            })

        final_df = pd.concat([final_df, temp_df], ignore_index=True)

    # Filtrer les capteurs en fonction de fibre
    final_df = final_df[
        ~final_df['MxB'].isin([f'M{i}{fibre}' for i in list(range(1, 9)) + list(range(36, 46))]) &  # Retirer M1x-M8x, M38x-M45x
        ~final_df['MxB'].isin([f'F{i}{fibre}' for i in list(range(1, 11)) + list(range(38, 46))])    # Retirer F1x-F8x, F38x-F45x
    ]

    # Décaler les positions X des capteurs FxD (ou FxB)
    final_df.loc[final_df['MxB'].str.startswith('F'), 'X'] += -1000

    return final_df

# Générer les plages de temps de 23:01:00 à 23:03:00 avec un décalage de 1 seconde
time_ranges = []
start_time = pd.to_datetime("06-25-24 23:01:25")
for i in range(15):  # 121 secondes de 23:01:00 à 23:03:00
    start_time_str = (start_time + timedelta(seconds=i))
    end_time_str = (start_time + timedelta(seconds=i + 60))
    time_ranges.append((start_time_str, end_time_str))



#------------------------------ Avec simulateur tableau de bord

  
# Fonction pour calculer l'angle entre deux pentes
image_folder = 'temp_images'
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

orange_condition_counter = 0

# Boucle sur les plages horaires
for frame, (start_time, end_time)    in enumerate(time_ranges): 
    final_df = process_data(start_time, end_time,fibre="B")
    
    seuil = 1
    filtered_df = final_df[final_df['Variation_t'] > seuil]
    fil_df = final_df[final_df['Variation_t'] < 0]

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
        cbar.set_label('Δ')
        plt.title(f'{end_time}')
        
        plt.savefig(f'{image_folder}/image_{frame}.png')
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
        r2l=model_left.score(X_left, y_left)

        # Régression pour la partie droite
        X_right = right_df['X'].values.reshape(-1, 1)
        y_right = right_df['Y'].values
        model_right = LinearRegression()
        model_right.fit(X_right, y_right)
        y_right_pred = model_right.predict(X_right)
        slope_right = model_right.coef_[0]
        r2r = model_right.score(X_right, y_right)
        
        
        
        
         
        L = final_df[(final_df['X'] >= int(X_left.min())) & (final_df['X'] <= int(X_left.max()))]['X'].unique()
        L = list(L)
    
        R = final_df[(final_df['X'] >= int(X_right.min())) & (final_df['X'] <= int(X_right.max()))]['X'].unique()
        R = list(R)
    
        
        
        
        variations_before_peak_L = []
        variations_after_peak_L = []
        variations_before_peak_R = []
        variations_after_peak_R = []

# Boucler sur les valeurs de X et tracer les courbes avec des décalages progressifs pour Y (loop for L)
        for i, x in enumerate(L):
            df_x = final_df[final_df['X'] == x]
            pic=df_x.loc[df_x['Variation_t'].idxmax()]
    # Split the data based on the peak
            df_before = df_x[df_x['Y'] <= pic['Y']-500]
            df_after = df_x[df_x['Y'] > pic['Y']+500]

    # Append values to lists for mean calculations later
            variations_before_peak_L.extend(df_before['Variation_t'].dropna())
            variations_after_peak_L.extend(df_after['Variation_t'].dropna())

    # Plotting
            if x*10 % 10 == 5:
                y_offset= slope_left*(i*50)-228.5
            else:
                y_offset = slope_left * (i * 50)
         
# Repeat the same logic for the R loop
        for i, x in enumerate(R):
            df_x = final_df[final_df['X'] == x] 
            pic=df_x.loc[df_x['Variation_t'].idxmax()]
# Split the data based on the peak
            df_before = df_x[df_x['Y'] <= pic['Y']-500]
            df_after = df_x[df_x['Y'] > pic['Y']+500]

    # Append values to lists for mean calculations later
            variations_before_peak_R.extend(df_before['Variation_t'].dropna())
            variations_after_peak_R.extend(df_after['Variation_t'].dropna())
    
    # Plotting
            if x*10 % 10==5:
                y_offset=slope_right * (i * 50)+228.5
            else:
                y_offset = slope_right * (i * 50)
                
# Calculate and display mean values
        mbL = np.mean(variations_before_peak_L) if variations_before_peak_L else np.nan
        maL = np.mean(variations_after_peak_L) if variations_after_peak_L else np.nan
        mbR = np.mean(variations_before_peak_R) if variations_before_peak_R else np.nan
        maR = np.mean(variations_after_peak_R) if variations_after_peak_R else np.nan
        
        # 5. Déterminer la couleur de fond en fonction des conditions
        background_color = 'green'  # Par défaut
        
        #distance de la base 
        dist = 500  # Ajuste cette valeur selon ce que tu considères proche

        # Filtrer les points proches de base_X en X
        close_high = filtered_df[(filtered_df['X'] >= base_X - dist) & (filtered_df['X'] <= base_X + dist)]
        
        

        # Vérification des pentes opposées et de l'angle
        if  -10 <= slope_left <= -2 and 2 <= slope_right<= 10 and len(close_high)>20 and r2l>0.5 and r2r>0.5 and (mbL>maL or mbR>maR):
        
                orange_condition_counter += 1
                if orange_condition_counter > 5:
                    background_color = 'red'
                    print(f"Alarme rouge à {end_time.time()}")
                else:
                    background_color = 'orange'
                    print(f"Alarme orange à {end_time.time()}")
         # Réinitialiser le compteur si la condition n'est plus remplie
        else:
            orange_condition_counter = 0
            
            
            
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
        
        plt.xlim([300, 3600])  
        plt.ylim([-100, 6300])

        cbar = plt.colorbar(scatter)
        cbar.set_label('Δ')
        plt.title(f'{end_time}')

        plt.savefig(f'{image_folder}/image_{frame}.png')
        plt.show()
        
images = []
for frame in range(len(time_ranges)):
    img_path = f'{image_folder}/image_{frame}.png'
    images.append(Image.open(img_path))
    
# Sauvegarder l'animation
gif_path = 'temperature_variation2.gif'
images[0].save(gif_path, save_all=True, append_images=images[1:], duration=1000, loop=1)

# Supprimer les images temporaires si nécessaire
for img in os.listdir(image_folder):
    os.remove(os.path.join(image_folder, img))
os.rmdir(image_folder)