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

## Part 2: 
