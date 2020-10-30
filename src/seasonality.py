## Part 0: Loading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

store = pd.HDFStore('../data/power_clean.h5')
df_train = store['df_train']
df_test = store['df_test']
store.close()

## Part 1: Formatting Data
df_train.set_index('DateTime', inplace=True)
df_train_D = df_train.resample('D').mean().iloc[:, 0]
df_train_H = df_train.resample('H').mean().iloc[:, 0]

df_train.index.dayofyear
