## Part 0: Loading
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

import pipeline

store = pd.HDFStore('../data/power_clean.h5')
pipe = pipeline.MyPipeline(store, [int(1e5), int(1e5)])
pipe.pre_fit(60)
data_out = pipe.pipe_full.transform(pipe.data)
store.close()

print("Finished copying data")

Y = data_out[:, 0].copy()
df_Y = pd.DataFrame(Y)

del data_out
store.close()

plt.clf()
plt.plot(Y)
plt.savefig("../pics/0_original.png")

## Part 1: Autocorrelation
# Hours
hours = Y.shape[0]//60
Y_hours = np.zeros((hours, 1))

for hour in range(hours):
    idx1 = 60*hour
    idx2 = 60*(hour+1)
    Y_hours[hour] = Y[idx1:idx2].mean()

plt.clf()
pd.plotting.autocorrelation_plot(Y_hours)
plt.title("Autocorrelation of Hours")
plt.savefig("../pics/ac_hours.png")

# Days
days = Y.shape[0]//(60*24)

Y_days = np.zeros((days, 1))

for day in range(days):
    idx1 = 24*day
    idx2 = 24*(day+1)
    Y_days[day] = Y_hours[idx1:idx2].mean()

plt.clf()
pd.plotting.autocorrelation_plot(Y_days)
plt.title("Autocorrelation of Days")
plt.savefig("../pics/ac_days.png")

## Part 2: Moving Average
window_size = 60
df_Y_window = df_Y.rolling(window=window_size)

df_Y_ma = pd.concat([df_Y, df_Y_window.mean()], axis=1)
df_Y_ma.columns = (["original", "window=" + str(window_size)])

# RMSE
df_Y_ma.iloc[:, 0] = df_Y_ma.iloc[:, 0].shift(-window_size)
Y_ma = df_Y_ma.dropna().to_numpy()
rmse_ma = ((Y_ma[:, 0] - Y_ma[:, 1])**2).mean()**0.5
rmse_ma = pipe.reverse_minmax(rmse_ma)[0][0]
print(f"rmse_ma = ", rmse_ma)

# Plotting
plt.clf()
plt.plot(df_Y_ma.iloc[:60*24*5, :].dropna())
plt.title("Moving Average Prediction 5 days")
plt.legend(df_Y_ma.columns)
plt.show()
plt.savefig("../pics/ma.png")

## Part 3: Seasonality Additive
result = seasonal_decompose(Y_hours, model='additive', period=24*365)

result.plot()
plt.savefig("../pics/seasonal_year.png")
plt.show()

a = df_Y.iloc[:, 0] == 0

Y_ma = df_Y_ma.iloc[:, 1].dropna()
result = seasonal_decompose(Y_ma[::60], model='additive', period=24*365)

result.plot()
plt.savefig("../pics/seasonal_year_ma.png")
plt.show()

## Part 4: Removing Seasonality
months = Y_hours.shape[0]//(24*30)
Y_months = []

for month in range(months):
    idx1 = month*24*30
    idx2 = (month+1)*24*30
    Y_months.append(Y_hours[idx1:idx2].mean())
    
Y_months = np.array(Y_months)

Y_months_avg = np.zeros(12)
for month in range(12):
    Y_months_avg[month] = Y_months[month::12].mean()

Y_hours_noseason = np.zeros(Y_hours.shape[0])

for hour in range(Y_hours.shape[0]):
    idx_hour = hour%8760
    idx_month = np.floor(idx_hour/(24*30.5))
    idx_month = int(idx_month)

    Y_hours_noseason[hour] = Y_hours[hour] - Y_months_avg[idx_month]

plt.plot(Y_hours_noseason)
plt.savefig("../pics/seasonality_no.png")

## Part 5: AR
from statsmodels.tsa.ar_model import AR

model = AR(Y_hours)
model_fit = model.fit()
lag = model_fit.k_ar #lag
params = model_fit.params

Y_hours_output = Y_hours[lag*2:]
Y_hours_input = Y_hours

Y_hours_input.shape
predictions = []
for t in range(Y_hours_input.shape[0]-lag):
    yhat = params[0]
    for i, param in enumerate(params[1:]):
        yhat = yhat + param*Y_hours_input[i+t, 0]

    predictions.append(yhat)


predictions = np.array(predictions)
predictions = predictions[lag:]
predictions = predictions.reshape(-1, 1)
predictions.shape
Y_hours_output.shape

rmse_ar = ((Y_hours_output-predictions)**2).mean()**0.5
rmse_ar = pipe.reverse_minmax(rmse_ar)[0][0]
print(f"rmse_ma = ", rmse_ar)

## Part 6: ARMA
from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(Y_hours, order=(5, 0, 1))
model_fit = model.fit(disp=0)

print(model_fit.summary())
