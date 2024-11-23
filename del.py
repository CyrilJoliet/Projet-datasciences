import pandas as pd
from datetime import timedelta
import warnings
from fun import process_data, plot, linear, temp

warnings.filterwarnings("ignore")

# Import DATA
df = pd.read_csv("C:/Users/abdel/Desktop/Git/Project_EBDS/AMDG - Sequence STR1-S-2024-06-25-14H21.csv", sep=';')

# Combine DATE and TIME into DATETIME column
df['DATETIME'] = pd.to_datetime(
    df['DATE'] + ' ' + df['TIME'], 
    format='%m/%d/%y %H:%M:%S'
)


# Print first value
# print(df['DATETIME'][0])
