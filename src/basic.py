## Part 0: Loading Data
import pandas as pd
import numpy as np

data1 = pd.read_csv('../data/power_cons1.txt', sep=";")
data2 = pd.read_csv('../data/power_cons2.txt', sep=";", header=None)

data2.columns = data1.columns
data = pd.concat([data1, data2])

del data1; del data2

## Part 1: Fixing Data
data.head()
data.tail()
data.info()
data['Date'] = pd.to_datetime(data['Date'])
data['Time'] = pd.to_datetime(data['Time'])
