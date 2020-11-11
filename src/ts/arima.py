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


## Part 1: ARIMA
from statsmodels.tsa.arima_model import ARIMA

data_in = df_train.iloc[:,0].resample("D").mean().dropna().values
model = ARIMA(data_in, order=(1,1,1))
model_fit = model.fit(disp=0)
print(model_fit.summary())

model_fit.resid
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

model_fit.plot_predict(dynamic=False)
plt.show()
