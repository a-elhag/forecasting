## Part 0: Loading
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pipeline

store = pd.HDFStore('../data/power_clean.h5')
pipe = pipeline.MyPipeline(store, [int(1e5), int(1e5)])
pipe.pre_fit(60)
data_out = pipe.pipe_full.transform(pipe.data)
store.close()

Y = data_out[:, 0].copy()
df_Y = pd.DataFrame(Y)
del data_out

print("Finished copying data")
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

## Part 3:
