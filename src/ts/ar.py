## Part 0: Loading
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

# Stats Models
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.stattools import durbin_watson

store = pd.HDFStore('../../data/power_clean.h5')
df_train = store['df_train']
df_test = store['df_test']
store.close()
df_train.set_index('DateTime', inplace=True)
df_test.set_index('DateTime', inplace=True)

idx = df_train.index >= '2007'
df_train = df_train[idx]

df_train = df_train.iloc[:, 0]
df_test = df_test.iloc[:, 0]

## Part 1: AR
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

lag_amount = 2
model = AutoReg(df_train.values, lags=lag_amount)
model_fit = model.fit()
coef = model_fit.params
predictions = model_fit.predict(
    start = 0, end = df_train.shape[0],
    dynamic = False)

predict = np.zeros((df_train.shape[0]-lag_amount+1))

for lag in range(1,lag_amount+1):
    if lag_amount == 1:
        df_train_lag = df_train.dropna().values
        predict = predict + df_train_lag*coef[lag]
        break

    if lag == 1:
        df_train_lag = df_train[lag_amount-1:].dropna().values
    elif lag == lag_amount+1:
        df_train_lag = df_train[:-lag_amount+1].dropna().values
    else:
        idx1 = -lag + lag_amount
        idx2 = -lag + 1
        df_train_lag = df_train[idx1:idx2].dropna().values

    predict = predict + df_train_lag*coef[lag]

predict = predict + coef[0]
error = (predict-predictions).sum()
print(error)
predictions.shape
df_train.shape

if lag_amount==1:
    rmse = np.sqrt(mean_squared_error(predict, df_train.values))
elif lag_amount==0:
    rmse = np.sqrt(mean_squared_error(predict[:-1], df_train.values))
else:
    idx1 = (lag_amount-1)*2
    idx2 = -(lag_amount-1)

    rmse = np.sqrt(mean_squared_error(predict[:idx2], df_train.values[idx1:]))
print(rmse)

plt.plot(predict[:idx2], label="predict")
plt.plot(df_train.values[idx1:], label="original")
plt.legend()
plt.grid()
plt.show()

predict[:idx2].shape
df_train.values[idx1:].shape

## Part 2: 
