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

lag_amount = 4
model = AutoReg(df_train.values, lags=lag_amount)
model_fit = model.fit()
coef = model_fit.params

train_lag1 = df_train[3:].dropna().values
train_lag2 = df_train[2:-1].dropna().values
train_lag3 = df_train[1:-2].dropna().values
train_lag4 = df_train[:-3].dropna().values

predict = train_lag4*coef[4] + train_lag3*coef[3] + train_lag2*coef[2] + train_lag1*coef[1] + coef[0]

predictions = model_fit.predict(
    start = 0, end = df_train.shape[0],
    dynamic = False)

error = (predictions-predict).sum()
print(error)


df_train[3:]
df_train[2:-1]

## Part 2: Predictions
predictions = model_fit.predict(
    start=df_train.shape[0], end = df_train.shape[0] + df_test.shape[0] -1,
    dynamic = True)

rmse = np.sqrt(mean_squared_error(df_test, predictions))

plt.plot(df_test.values[:1000], label="Original")
plt.plot(predictions[:1000], label="Predictions")
plt.grid()
plt.legend()
plt.show()

## Part 2: 
