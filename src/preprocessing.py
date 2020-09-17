"""
Note:
     On github, the max size for a document is 100MB, but we want to use the HD5 file type
     because it is wicked fast. That means, that you have to run this file once to have it set
     up. Because the output of this code is added to .gitignore
"""

## ==> Part 0: Setting Up
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

## ==> Part 1: Preprocessing
## Part 1a: Fixing Data and Time
df_data['Date'] = pd.to_datetime(df_data['Date'])
df_data['Time'] = pd.to_datetime(df_data['Time'], format="%H:%M:%S")

# df_data['Year'] = [d.year for d in df_data['Date']]
# df_data['Month'] = [d.month for d in df_data['Date']]
# df_data['Day'] = [d.day for d in df_data['Date']]
# 
# df_data['Hour'] = [d.hour for d in df_data['Time']]
# df_data['Minute'] = [d.minute for d in df_data['Time']]
# 
# df_data = df_data.drop(['Date', 'Time'], axis=1)

## Part 1b: Converting Objects to Float
for col in range(6):
    df_data.iloc[:, col+2] = pd.to_numeric(df_data.iloc[:, col+2], errors='coerce')

## Part 1c: Removing missing values
df_data.dropna(axis=0, how='any', inplace=True)

## Part 1d: Splitting into Testing and Training
Test = len(df_data[df_data.Date < "2010-01-01"])
Total = len(df_data)
# Test/Total = 78 %

df_test = df_data[df_data.Date >= "2010-01-01"]
df_train = df_data[df_data.Date < "2010-01-01"]
del df_data

## Part 1e: Saving the data
store = pd.HDFStore('../data/power_clean.h5')
store['df_train'] = df_train
store['df_test'] = df_test
store.close()
