
import pandas as pd
import math 
import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.colors as mcolors
from datetime import timedelta
from sklearn.linear_model import LinearRegression

#________________________________ vous devez changer le chemin ___________________________________________#
df = pd.read_csv('C:/Users/abdel\Desktop/Project_EBDS/AMDG - Sequence STR1-S-2024-06-25-14H21.csv', sep=';')

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

#___________________________________________ ploting _____________________________________#

import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.linear_model import LinearRegression

# Function to update plot
def update_plot(final_df, start_time, end_time, canvas, ax, cbar):
    ax.clear()  # Clear previous plot

    seuil = 0.6
    filtered_df = final_df[final_df['Variation_t'] > seuil]

    if filtered_df.empty:
        norm = mcolors.TwoSlopeNorm(vmin=-1., vmax=2, vcenter=0)
        cmap = plt.get_cmap('coolwarm')  

        scatter = ax.scatter(final_df['X'], final_df['Y'], c=final_df['Variation_t'], s=25, cmap=cmap, norm=norm)
        ax.set_xlim([300, 3600])  # Set X axis limits
        ax.set_ylim([-100, 6300])  # Set Y axis limits

        if cbar is None:  # Create the color bar only once
            cbar = plt.colorbar(scatter, ax=ax)
        else:
            cbar.update_normal(scatter)  # Update existing color bar

        cbar.set_label('Variation de Température')
        ax.set_title(f'Variation de Température de {start_time.time()} à {end_time}')
        
        ax.set_facecolor('lightgray')  # Set plot area to light gray
        canvas.figure.set_facecolor('lightgreen')  # Set figure background to green
    else:
        min_Y_point = filtered_df.loc[filtered_df['Y'].idxmin()]
        base_X = min_Y_point['X']
        base_Y = min_Y_point['Y']

        left_df = filtered_df[filtered_df['X'] <= base_X]
        right_df = filtered_df[filtered_df['X'] >= base_X]

        X_left = left_df['X'].values.reshape(-1, 1)
        y_left = left_df['Y'].values
        model_left = LinearRegression()
        model_left.fit(X_left, y_left)
        y_left_pred = model_left.predict(X_left)

        X_right = right_df['X'].values.reshape(-1, 1)
        y_right = right_df['Y'].values
        model_right = LinearRegression()
        model_right.fit(X_right, y_right)
        y_right_pred = model_right.predict(X_right)

        norm = mcolors.TwoSlopeNorm(vmin=-1., vmax=2, vcenter=0)
        cmap = plt.get_cmap('coolwarm')
        
        # --- Extract Slopes ---
        left_slope = model_left.coef_[0]
        right_slope = model_right.coef_[0]

        # Calculate the angle in radians
        angle_radians = math.atan(abs((left_slope - right_slope) / (1 + left_slope * right_slope)))
        
        # Ensure the angle is the smaller "upwards" angle
        if angle_radians > math.pi / 2:  # If the angle is greater than 90 degrees
            angle_radians = math.pi - angle_radians  # Subtract from 180 degrees (pi radians)
        
        angle_degrees = math.degrees(angle_radians)  # Convert to degrees if needed


        ax.set_facecolor('lightgray')  # Set plot area to light gray
        canvas.figure.set_facecolor('lightgreen')  # Set figure background to green

        # --- Set Background Color Based on Angle ---
        if 25 <= angle_degrees <= 60:
            canvas.figure.set_facecolor('red')  # Set figure background to red
        else:
            canvas.figure.set_facecolor('lightgreen')  # Set figure background to green

        # --- Display Angle in Bottom Middle ---
        angle_text = f'Angle: {angle_degrees:.2f}°'
        ax.text(0.5, 0.05, angle_text, 
                transform=ax.transAxes, 
                horizontalalignment='center', 
                verticalalignment='center', 
                bbox=dict(facecolor='white', alpha=0.8)) # Add a box around the text


        scatter = ax.scatter(final_df['X'], final_df['Y'], c=final_df['Variation_t'], s=25, cmap=cmap, norm=norm)
        ax.plot(X_left, y_left_pred, color='black', label=f'Régression gauche\nPente: {model_left.coef_[0]:.2f}')
        ax.plot(X_right, y_right_pred, color='black', label=f'Régression droite\nPente: {model_right.coef_[0]:.2f}')
        
        ax.set_xlim([300, 3550])  # Set X axis limits
        ax.set_ylim([-100, 6200])  # Set Y axis limits

        if cbar is None:  # Create the color bar only once
            cbar = plt.colorbar(scatter, ax=ax)
        else:
            cbar.update_normal(scatter)  # Update existing color bar

        cbar.set_label('Variation de Température (°C)')
        ax.set_title(f'Variation de Température de {start_time.time()} à {end_time.time()}')
    
    canvas.draw()  # Redraw the updated plot
    return cbar  # Return the color bar for further updates
# Main function to create Tkinter window and embed the plot
def run_gui(time_ranges, process_data):
    # Create Tkinter window
    root = tk.Tk()
    root.title("Temperature Variation Plot")

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    cbar = None  # Initialize color bar variable

    # Loop through time ranges and update the plot
    for start_time, end_time in time_ranges:
        final_df = process_data(start_time, end_time)
        cbar = update_plot(final_df, start_time, end_time, canvas, ax, cbar)  # Pass and update the color bar
        root.update()  # Update the Tkinter window to reflect changes
        root.after(100)  # Pause for 1 second between updates

    root.mainloop()  # Start the Tkinter main loop


# Example usage:
time_ranges = []
start_time = pd.to_datetime("25-06-24 23:01:00")
for i in range(128):  # 121 secondes de 23:01:00 à 23:03:00
    start_time_str = (start_time + timedelta(seconds=i))
    end_time_str = (start_time + timedelta(seconds=i + 60))
    time_ranges.append((start_time_str, end_time_str))
run_gui(time_ranges, process_data)
