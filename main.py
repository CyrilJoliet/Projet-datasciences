# importing the packages
import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
from fun import process_data,plot,linear,temp
warnings.filterwarnings("ignore")


# Importing the DATA
df = pd.read_csv('Industeel-S2024-02-12.csv', sep=';')

# Transform the date in datetime format
df['DATETIME'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'],format='%y/%m/%d %H:%M:%S')
df['TIME'] = [(df['DATETIME'][0]+ timedelta(seconds=i)) for i in range(len(df))]


#  Generate 60 sec timeframes from start_time incremented by 1sec 

starting_time = "02-12-24 23:15:10"         # this is the starting time
Num_frames = 150                             # this is number of frames to analyze

time_ranges = []
start_time = pd.to_datetime(starting_time)   
for i in range(Num_frames):                               
    start_time_str = (start_time + timedelta(seconds=i))
    end_time_str = (start_time + timedelta(seconds=i + 60))
    time_ranges.append((start_time_str, end_time_str))


orange_condition_counter = 0

# Loop on the time frames
for start_time, end_time  in time_ranges: 
    data = process_data(df,start_time, end_time, captors='A',n_capt= 42)    # captors == A or D  |  n_capt == 42 or 45
    final_df=data[0]
    Speed=data[1]
    high = final_df[final_df['Variation_t'] > 1 ]
    Index = round(min( len(high) / 100 , 0.35),2)

    # Check if there is more than 30 high value (variation_t that exceed 1)
    if  len(high)<30:
        plot(final_df,end_time,"green",Index = round(Index,2))

    else: 
        #Take maximum values 
        filtered_df=high.loc[high.groupby('X')['Variation_t'].idxmax()]
        
        #Find lower point as base of the V with at least 20 high values around
        points_around_min = []
        i=0
        while len(points_around_min) < 10:
            
            sorted_df = filtered_df.sort_values(by='Y').reset_index(drop=True)
            min_Y_point = filtered_df.loc[(filtered_df == sorted_df.iloc[i]).all(axis=1)].iloc[0]
            points_around_min = high[ (high['X'] >= min_Y_point['X']-150) & (high['X'] <= min_Y_point['X']+150) &(high['Y'] <= min_Y_point['Y'] +50 )&(high['Y']>=min_Y_point['Y']-50)]
            if i==len(sorted_df)-1:
                break
            i+=1
            
            
        #Separate data for left and right regression
        left_df = filtered_df[filtered_df['X'] <= min_Y_point['X']]
        right_df = filtered_df[filtered_df['X'] >= min_Y_point['X']]
        
        #Check if more than 3 value left and right of the base of V
        #Calculate linear regression on both part
        if len(left_df)>3 and len(right_df)>3:
            Index += 0.1
            # if alert is confirmed, stop calculating regressions
            if orange_condition_counter<10:
            
                Left=linear(left_df['X'],left_df['Y'])
                Right=linear(right_df['X'],right_df['Y'])
            else: 
                Left = (Left[0], Left[1] - Speed, Left[2], Left[3])
                Right=(Right[0],Right[1]-Speed,Right[2],Right[3])
        
            # getting the X of the sensors
            L=list(left_df['X'])
            R=list(right_df['X'])
            
            #Check if temperature is decreasing or increasing
            variation_L= temp(L, final_df)
            variation_R= temp(R,final_df)
         
            Td = final_df[(final_df['X'] > int(left_df['X'].min()+100)) & (final_df['X'] < int(right_df['X'].max()-100))&(final_df['Y'] > (final_df['Y'].max()-50))]
            T=Td['Variation_t'].mean()

        # Verification on regression slope and R2 and on temperature variation
        # raise alert if conditions are respected
        # change color, calculate the index and add the regression lines on the visualisation

            if  -15 <= (Left[2] <= 0 or 0 <= Right[2]<= 15) and (Left[3]>0.8 or Right[3]>0.8) and (0>variation_L or 0>variation_R):
        
                    orange_condition_counter += 1
                    if orange_condition_counter >5 and T>0:
                        background_color = 'orange'
                        Index = 0.6
                        
                    elif orange_condition_counter > 5 :
                        background_color = 'red'
                        Index = 1
                        print(f"Alert red à {end_time.time()}")
                    else:
                            background_color = 'orange'
                            Index += round(min(orange_condition_counter/10, 0.6),2)
                            print(f"Alert orange à {end_time.time()}")
                
            else:
                
                background_color='green'
            
            Index = min(Index, 1)
            plot(final_df,end_time,background_color,Left[0],Left[1],Right[0],Right[1],Left[3],Right[3],Index=round(Index,2))
           
        else :
            plot(final_df,end_time,'green',Index = Index)
 
