## Part 0a: Loading Data
import pandas as pd
import numpy as np

data1 = pd.read_csv('../data/power_cons1.txt', sep=";")
data2 = pd.read_csv('../data/power_cons2.txt', sep=";", header=None)

data2.columns = data1.columns
df_data = pd.concat([data1, data2])

del data1; del data2

## Part 0b: Exploration of Dataframe
df_data.head()
df_data.tail()
df_data.info()

## Part 1: Fixing Data
df_data['Date'] = pd.to_datetime(df_data['Date'])
df_data['Time'] = pd.to_datetime(df_data['Time'], format="%H:%M:%S")

df_data['Year'] = [d.year for d in df_data['Date']]
df_data['Month'] = [d.month for d in df_data['Date']]
df_data['Day'] = [d.day for d in df_data['Date']]

df_data['Hour'] = [d.hour for d in df_data['Time']]
df_data['Minute'] = [d.minute for d in df_data['Time']]

df_data = df_data.drop(['Date', 'Time'], axis=1)

df_data.head()
