# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 17:36:08 2024

@author: cyril
"""
import numpy as np
import pandas as pd
import math
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# Set the data history on X Y position where X is the positions off captors and Y is calculated with the speed and the time 
# df = data 
# start_time , end_time of the time frame
# captors = name of the captor (A,B,C,D,..)
#n_captors= number of captors by face (industeel=42)

def process_data(df,start_time, end_time, captors,n_capt):
        
    # Take data on the timeframe
    time = df[(df['TIME'] >= start_time) & (df['TIME'] <= end_time)]
    
    # Take only usefull columns from the dataframe
    data = time[['TIME', 'WIDTH','SPEED'] +[f'M{i}{captors}' for i in range(1, n_capt+1)] + [f'F{i}{captors}' for i in range(1, n_capt+1)]]

    # Calculate tempreature variation (difference between Temperature at t and t-1 )
    colonnes_temp = [f'M{i}{captors}' for i in range(1, n_capt+1)] + [f'F{i}{captors}' for i in range(1, n_capt+1)]
    for col in colonnes_temp:
        data[f'variation_{col}'] = data[col].diff()
    
    # number of second of the time frame
    n_mesures = int((end_time-start_time).total_seconds())+1

    # Create grid with X Y positions
    Speed=data['SPEED'].mean()/6
    Y = np.arange(n_mesures)*Speed
    X = [i * 50 for i in range(n_capt)] + [(i * 50 + 228.5 + 50 * (n_capt-1)) for i in range(n_capt)] 
    final_df = pd.DataFrame(columns=['X', 'Y', 'Variation_t'])

    for i in range(n_capt*2):
        if i < n_capt:
            temp_df = pd.DataFrame({
                'X': [X[i]] * n_mesures,             
                'Y': Y,                              
                'Variation_t': data[f'variation_M{i+1}{captors}'],  
                'MxB': [f'M{i+1}{captors}'] * n_mesures
            })
        else:
            temp_df = pd.DataFrame({
                'X': [X[i]] * n_mesures,             
                'Y': Y,                              
                'Variation_t': data[f'variation_F{i-n_capt+1}{captors}'],  
                'MxB': [f'F{i-n_capt+1}{captors}'] * n_mesures  
            })

        final_df = pd.concat([final_df, temp_df], ignore_index=True)
        
    # Eject captors outside the width
    ce=int(data["WIDTH"].max()/50)-2
    p=math.ceil((n_capt-ce)/2)
    final_df = final_df[
        ~final_df['MxB'].isin([f'M{i}{captors}' for i in list(range(1, p+1)) + list(range(n_capt+1-p, n_capt+1))]) &  # Retirer M1x-M8x, M38x-M45x
        ~final_df['MxB'].isin([f'F{i}{captors}' for i in list(range(1, p+1)) + list(range(n_capt+1-p, n_capt+1))])    # Retirer F1x-F8x, F38x-F45x
    ]
    # Move position of active captors
    final_df.loc[final_df['MxB'].str.startswith('F'), 'X'] += -p*100

    return final_df,Speed

#Visualisation of temerature variations

def plot(df,end_time,background_color,X_left= [],y_left_pred=[],X_right=[],y_right_pred=[],r2l=0,r2r=0) :
    norm = mcolors.TwoSlopeNorm(vmin=-1., vmax=2, vcenter=0)
    cmap = plt.get_cmap('coolwarm')  

    fig, ax = plt.subplots(figsize=(10, 6)) 
    
    scatter = plt.scatter(df['X'], df['Y'], c=df['Variation_t'], s=25, cmap=cmap, norm=norm)
    if len(X_left) > 0 and len(y_left_pred) > 0 and len(X_right) > 0 and len(y_right_pred) > 0:  # Vérifie si les valeurs ne sont pas nulles ou vides
        ax.plot(X_left, y_left_pred, color='black')
        ax.plot(X_right, y_right_pred, color='black')
        ax.text(0.9, 0.02, f'$R^{{2}}_{{\\text{{left}}}}={r2l:.2f}$\n$R^{{2}}_{{\\text{{right}}}}={r2r:.2f}$', transform=ax.transAxes, 
        fontsize=12, verticalalignment='bottom', horizontalalignment='center', 
        bbox=dict(boxstyle="round", facecolor="white", alpha=1))
    plt.xlim([300, 3600])  

    fig.patch.set_facecolor(background_color)
    
    cmap = plt.get_cmap('coolwarm')

    cbar = plt.colorbar(scatter)
    cbar.set_label('Δ')
    plt.title(f'{end_time}')

    plt.show()

#Linear regressions    
    
def linear(x,y):
    X = x.values.reshape(-1, 1)
    Y = y.values
    model = LinearRegression()
    model.fit(X, Y)
    y_pred = model.predict(X)
    slope = model.coef_[0]  
    r2 = model.score(X, Y)
    return X,y_pred, slope, r2

#Calculate mean variations after the V

def temp(L,df):
    variations_after_peak = []
    for i,x in enumerate(L):
        dfx=df[df['X']==x]
        pic=dfx.loc[dfx['Variation_t'].idxmax()]
        df_after = dfx[dfx['Y'] > pic['Y']+100]
        variations_after_peak.extend(df_after['Variation_t'].dropna())
    vap= np.mean(variations_after_peak)
    return vap
    
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
