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

## Part 1: Autocorrelation
days = Y.shape[0]//60
Y_days = np.zeros((days, 1))

for day in range(days):
    idx1 = 60*day
    idx2 = 60*(day+1)
    Y_days[day] = Y[idx1:idx2].mean()

df_days = pd.DataFrame(Y_days)

pd.plotting.autocorrelation_plot(Y_days)
plt.title("Autocorrelation of Days")
plt.show()

## Part 2: 
