

import pandas as pd
import math
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
df= pd.read_csv(r"C:\Users\cyril\OneDrive\Documents\cours\M2\DATASCIENCES\wetransfer_fichiers-industeel_2024-11-04_1143\Industeel - Sequence STR1-S-2024-02-12-22H23.csv", sep=';')
df['DATETIME'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'],format='%y/%m/%d %H:%M:%S')

# Accéder au premier élément de la colonne 'DATETIME'
start = df['DATETIME'][0]
df['TIME'] = [(start + timedelta(seconds=i)) for i in range(len(df))]


# Fonction pour traiter les données
def process_data(start_time, end_time, fibre='A',n_capt=42):
    # Vérification de la validité de l'argument fibre
    if fibre not in ['B', 'D','A','C']:
        raise ValueError("Le paramètre 'fibre' doit être 'B' ou 'D'")
    
    # Filtrer les données sur la période souhaitée
    collage_time = df[(df['TIME'] >= start_time) & (df['TIME'] <= end_time)]
    
    # Colonnes à garder, adaptant MxB/FxB ou MxD/FxD selon fibre
    colonnes_a_garder = ['DATE', 'TIME', 'SPEED', 'WIDTH', 'LENGTH'] + \
                        [f'M{i}{fibre}' for i in range(1, n_capt+1)] + [f'F{i}{fibre}' for i in range(1, n_capt+1)]

    collage_fibre = collage_time[colonnes_a_garder]

    # Colonnes pour les variations de température (BxB ou DxD selon fibre)
    colonnes_temp = [f'M{i}{fibre}' for i in range(1, n_capt+1)] + [f'F{i}{fibre}' for i in range(1, n_capt+1)]

    # Calculer les variations de température
    for col in colonnes_temp:
        collage_fibre[f'variation_{col}'] = collage_fibre[col].diff()

    n_mesures = 61  # Nombre de mesures (secondes)
    

    # Coordonnées X et Y
    Y = np.arange(n_mesures) *  100 #collage_fibre["SPEED"].mean()
    X = [i * 50 for i in range(n_capt)] + [(i * 50 + 228.5 + 50 * (n_capt-1)) for i in range(n_capt)] 
    final_df = pd.DataFrame(columns=['X', 'Y', 'Variation_t'])

    for i in range(n_capt*2):
        if i < n_capt:
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
                'Variation_t': collage_fibre[f'variation_F{i-n_capt+1}{fibre}'],  # Variation pour F
                'MxB': [f'F{i-n_capt+1}{fibre}'] * n_mesures  # Nom du capteur F
            })

        final_df = pd.concat([final_df, temp_df], ignore_index=True)
        
        ce=int(collage_fibre["WIDTH"].max()/50)-2
        p=math.ceil((n_capt-ce)/2)
     
        
    # Filtrer les capteurs en fonction de fibre
    final_df = final_df[
        ~final_df['MxB'].isin([f'M{i}{fibre}' for i in list(range(1, p+1)) + list(range(n_capt+1-p, n_capt+1))]) &  # Retirer M1x-M8x, M38x-M45x
        ~final_df['MxB'].isin([f'F{i}{fibre}' for i in list(range(1, p+1)) + list(range(n_capt+1-p, n_capt+1))])    # Retirer F1x-F8x, F38x-F45x
    ]

    # Décaler les positions X des capteurs FxD (ou FxB)
    final_df.loc[final_df['MxB'].str.startswith('F'), 'X'] += -p*100

    return final_df

# Générer les plages de temps de 23:01:00 à 23:03:00 avec un décalage de 1 seconde
time_ranges = []
start_time = pd.to_datetime("02-12-24 23:29:15")
for i in range(10):  # 121 secondes de 23:01:00 à 23:03:00
    start_time_str = (start_time + timedelta(seconds=i))
    end_time_str = (start_time + timedelta(seconds=i + 60))
    time_ranges.append((start_time_str, end_time_str))



#------------------------------ Avec simulateur tableau de bord

image_folder = 'temp_images'
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

orange_condition_counter = 0

# Boucle sur les plages horaires
for frame, (start_time, end_time)    in enumerate(time_ranges): 
    final_df = process_data(start_time, end_time)
    
    seuil = 1
    max_indices = final_df.groupby('X')['Variation_t'].idxmax()

# Filter final_df to only keep the rows with the maximum 'Variation_t' for each 'X'
    filtered_df = final_df.loc[max_indices]
    filtered_df = filtered_df[filtered_df['Variation_t'] > seuil]
    fil=final_df[final_df['Variation_t']>seuil]
    
    # Vérifier s'il y a des points filtrés avant de continuer
    if filtered_df.empty or len(fil)<50:
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
        
        if len(left_df)>3 and len(right_df)>3:
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
        
        
            
            T = final_df[(final_df['X'] >= int(X_left.min())) & (final_df['X'] <= int(X_right.max()))&(final_df['Y'] == 6000)]
            
        
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
                df_before = df_x[df_x['Y'] <= pic['Y']-100]
                df_after = df_x[df_x['Y'] > pic['Y']+500]

        # Append values to lists for mean calculations later
                variations_before_peak_L.extend(df_before['Variation_t'].dropna())
                variations_after_peak_L.extend(df_after['Variation_t'].dropna())

# Repeat the same logic for the R loop
            for i, x in enumerate(R):
                df_x = final_df[final_df['X'] == x] 
                pic=df_x.loc[df_x['Variation_t'].idxmax()]
# Split the data based on the peak
                df_before = df_x[df_x['Y'] <= pic['Y']-100]
                df_after = df_x[df_x['Y'] > pic['Y']+500]

    # Append values to lists for mean calculations later
                variations_before_peak_R.extend(df_before['Variation_t'].dropna())
                variations_after_peak_R.extend(df_after['Variation_t'].dropna())

                
# Calculate and display mean values
            mbL = np.mean(variations_before_peak_L) if variations_before_peak_L else np.nan
            maL = np.mean(variations_after_peak_L) if variations_after_peak_L else np.nan
            mbR = np.mean(variations_before_peak_R) if variations_before_peak_R else np.nan
            maR = np.mean(variations_after_peak_R) if variations_after_peak_R else np.nan
        
        # 5. Déterminer la couleur de fond en fonction des conditions
            background_color = 'green'  # Par défaut
    
        

        # Vérification des pentes opposées et de l'angle
            if  -17 <= slope_left <= -2 and 2 <= slope_right<= 17 and r2l>0.8 and r2r>0.8 and (0>maL or 0>maR):
        
                    orange_condition_counter += 1
                    if orange_condition_counter >5 and T['Variation_t'].mean()>0:
                        
                        background_color = 'yellow'
                        print("DECOLAGE")
                    elif orange_condition_counter > 5 :
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
        else :
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