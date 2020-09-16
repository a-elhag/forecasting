## Part 0a: Loading Data
import pandas as pd
import numpy as np

data1 = pd.read_csv('../data/power_cons1.txt', sep=";")
data2 = pd.read_csv('../data/power_cons2.txt', sep=";", header=None)

data2.columns = data1.columns
df_data = pd.concat([data1, data2], ignore_index=True)

del data1; del data2

## Part 0b: Exploration of Dataframe
pd.set_option('display.float_format', lambda x: '%.2f' % x)

df_data.head()
df_data.tail()
df_data.info()
df_data.describe()

col_range = 5
for _ in range(int(np.ceil(len(df_data.columns)/col_range))):
    idx1 = _*col_range
    idx2 = idx1+col_range
    print(df_data.iloc[:, idx1:idx2].describe())

## Part 1: Fixing Data and Time
df_data['Date'] = pd.to_datetime(df_data['Date'])
df_data['Time'] = pd.to_datetime(df_data['Time'], format="%H:%M:%S")

df_data['Year'] = [d.year for d in df_data['Date']]
df_data['Month'] = [d.month for d in df_data['Date']]
df_data['Day'] = [d.day for d in df_data['Date']]

df_data['Hour'] = [d.hour for d in df_data['Time']]
df_data['Minute'] = [d.minute for d in df_data['Time']]

df_data = df_data.drop(['Date', 'Time'], axis=1)

## Part 1b: Converting Objects to Float
for col in range(6):
    df_data.iloc[:, col] = pd.to_numeric(df_data.iloc[:, col], errors='coerce')

