## Part 0: Loading
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

# Stats Models
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.stattools import durbin_watson

store = pd.HDFStore('../data/power_clean.h5')
df_train = store['df_train']
df_test = store['df_test']
store.close()
df_train.set_index('DateTime', inplace=True)
df_test.set_index('DateTime', inplace=True)

idx = df_train.index >= '2007'
df_train = df_train[idx]

# Part 1: Set up
df_train = df_train.iloc[:,0]

## Part 2: Integrated
from statsmodels.tsa.stattools import adfuller

# df_train.diff().diff() ==> 2 diffs!
result = adfuller(df_train.resample("D").mean().dropna().values)

print('ADF Statistic: ', result[0])
print('p-value: ', result[1])

## Part 3: Automatic Tests
from pmdarima.arima.utils import ndiffs

y = df_train.resample("H").mean().dropna().values

## Adf Test
ndiffs(y, test='adf')  # 2
# KPSS test
ndiffs(y, test='kpss')  # 0
# PP test:
ndiffs(y, test='pp')  # 2

