## Part 0: Loading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

store = pd.HDFStore('../data/power_clean.h5')
df_train = store['df_train']
df_test = store['df_test']
store.close()

df_train.set_index('DateTime', inplace=True)
df_test.set_index('DateTime', inplace=True)

## Part 1: Formatting Data
df_train_M = df_train.resample('M').mean().iloc[:, 0]
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

## Part 2: Making it into a function
def season(df, freq):
    df = df.resample(freq).mean().iloc[:,0]

    df_idx = {
        "MS": df.index.month,
        "W": pd.Int64Index(df.index.isocalendar().week), 
        "D": df.index.dayofyear,
        "H": df.index.hour,
    }

    df_idx[freq]
    df_idx_max = df_idx[freq].max()

    season = np.zeros((df_idx_max, 1))

    for t in range(1, df_idx_max+1):
        df_idx_t = df_idx[freq].isin([t])
        season[t-1] = np.nanmean(df.loc[df_idx_t].values)

    season = season/df.mean()
    return season 

season_M = season(df_train, "MS")
season_W = season(df_train, "W")
season_D = season(df_train, "D")
season_H = season(df_train, "H")

## Part 3: Residual
df = df_train.iloc[:, 0]

season_M_rep = season_M[df.index.month-1]
season_W_rep = season_W[pd.Int64Index(df.index.isocalendar().week)-1]
season_D_rep = season_D[df.index.dayofyear-1]
season_H_rep = season_H[df.index.hour-1]

data_in = df.values.reshape(-1,1)
season_M_rep.shape
season_W_rep.shape
season_D_rep.shape
season_H_rep.shape


fig, axs = plt.subplots(3)
axs[0].plot(data_in)
axs[0].set_title('Original')
axs[1].plot(season_D_rep*season_H_rep)
axs[1].set_title('Seasonality')
axs[2].plot(data_in/season_D_rep/season_H_rep)
axs[2].set_title('Residual')

fig.tight_layout()
plt.grid()
plt.show()


## Part 4: 

