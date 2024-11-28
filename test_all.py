# Importing the libraries
import pandas as pd
from datetime import timedelta
import warnings
from tqdm import tqdm
from fun import process_data,plot,linear,temp
warnings.filterwarnings("ignore")


# read csv 
df= pd.read_csv(r"C:\Users\cyril\OneDrive\Documents\cours\M2\DATASCIENCES\wetransfer_fichiers-industeel_2024-11-04_1143\Industeel - Sequence STR1-S-2024-02-12-22H23.csv", sep=';')

# Transform the date in datetime format
df['DATETIME'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'],format='%y/%m/%d %H:%M:%S') 
df['TIME'] = [(df['DATETIME'][0]+ timedelta(seconds=i)) for i in range(len(df))]


# Generate 60 sec timeframes from start_time incremented by 1sec
time_ranges = []
start_time = df["TIME"][0]
for i in range(len(df)-60):  
    start_time_str = (start_time + timedelta(seconds=i))
    end_time_str = (start_time + timedelta(seconds=i + 60))
    time_ranges.append((start_time_str, end_time_str))



orange_condition_counter = 0
red_alert = []
orange_alert = []


# Process on generate time frames
for start_time, end_time in tqdm(time_ranges, desc="Processing time ranges"): 
    data = process_data(df,start_time, end_time, captors='A',n_capt=42)
    final_df=data[0]
    Speed=data[1]
    high=final_df[final_df['Variation_t']>1]
    
    # Check if there is more than 30 high value (variation>1)
    if  len(high)<30:
        
        None
        
    else: 
        #Take maximum values 
        filtered_df=high.loc[high.groupby('X')['Variation_t'].idxmax()]
        
        #Find lower point as base off the V with at least 20 high values around
        points_around_min = []
        i=0
        while len(points_around_min)<10:
            
            sorted_df = filtered_df.sort_values(by='Y').reset_index(drop=True)
            min_Y_point = filtered_df.loc[(filtered_df == sorted_df.iloc[i]).all(axis=1)].iloc[0]
            points_around_min = high[ (high['X'] >= min_Y_point['X']-150) & (high['X'] <= min_Y_point['X']+150) &(high['Y'] <= min_Y_point['Y'] +50 )&(high['Y']>=min_Y_point['Y']-50)]
            if i==len(sorted_df)-1:
                break
            i+=1
            
            
        #Separate data for left and right regression
        left_df = filtered_df[filtered_df['X'] <= min_Y_point['X']]
        right_df = filtered_df[filtered_df['X'] >= min_Y_point['X']]
        
        #Check if more than 3 value in both part
        #Calculate linear regression on both part
        if len(left_df)>3 and len(right_df)>3:
            # if alert is confirmed, stop calculating regressions
            if orange_condition_counter<10:
            
                Left=linear(left_df['X'],left_df['Y'])
                Right=linear(right_df['X'],right_df['Y'])
            else: 
                Left = (Left[0], Left[1] - Speed, Left[2], Left[3])
                Right=(Right[0],Right[1]-Speed,Right[2],Right[3])
        
            L=list(left_df['X'])
           
            R=list(right_df['X'])
            
            #Check if temperature is decreasing or increasing
            
            variation_L= temp(L, final_df)
            
            variation_R= temp(R,final_df)
         
            Td = final_df[(final_df['X'] > int(left_df['X'].min()+100)) & (final_df['X'] < int(right_df['X'].max()-100))&(final_df['Y'] > (final_df['Y'].max()-50))]
            T=Td['Variation_t'].mean()
            
        # Verification on regression slope and R2 and on temperature variation
        #raise alert if conditions are respected
        #change color and ad regressions on visualisation
        
            if  -15 <= (Left[2] <= 0 and 0 <= Right[2]<= 15) and (Left[3]>0.8 and Right[3]>0.8) and (0>variation_L or 0>variation_R):
        
                    orange_condition_counter += 1
                    if orange_condition_counter >5 and T>0:
                        
                        background_color = 'orange'
                        
                    elif orange_condition_counter > 5 :
                        background_color = 'red'
                        print(f"Alert red à {end_time.time()}")
                        red_alert.append(end_time)
                    else:
                            background_color = 'orange'
                            print(f"Alert orange à {end_time.time()}")
                            orange_alert.append(end_time)
                    plot(final_df,end_time,background_color,Left[0],Left[1],Right[0],Right[1],Left[3],Right[3])             
            else:
                orange_condition_counter = 0
                background_color='green'
            
                    
           
        else :
            None
            
            
         
      

        
