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

df_train_D_idx = df_train_D.index.dayofyear.isin([365])
df_train_D.loc[df_train_D_idx]

season_D = np.zeros((366,1))
_season_D_mean = df_train_D.mean()
for day in range(1, 367):
    df_train_D_idx = df_train_D.index.dayofyear.isin([day])
    season_D[day-1] = df_train_D.loc[df_train_D_idx].values.mean()

season_D = season_D/_season_D_mean
plt.plot(np.tile(season_D, (3,1)))
plt.show()

