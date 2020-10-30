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

def plot_season(season, name):
    plt.clf()
    fig, axs = plt.subplots(3)
    axs[0].plot(data_in)
    axs[0].set_title('Original')
    axs[1].plot(season)
    axs[1].set_title('Seasonality ' + name)
    axs[2].plot(data_in/season)
    axs[2].set_title('Residual')

    fig.tight_layout()
    plt.grid()
    plt.show()

plot_season(season_M_rep, "Month")
plot_season(season_W_rep, "Week")
plot_season(season_D_rep, "Day")
plot_season(season_H_rep, "Hour")
plot_season(season_H_rep*season_D_rep, "Day+Hour")
plot_season(season_W_rep*season_H_rep*season_D_rep, "Day+Week+Hour")

## Part 4: Autocorrelation
from statsmodels.graphics.tsaplots import plot_acf
# plot_acf(df.dropna().values, lags=100)

def plot_season_ac(season, freq, name, season_flag = True):
    if season_flag:
        df = df_train.iloc[:, 0].div(season.reshape(-1))
    else: 
        df = df_train.iloc[:, 0]
    df = df.resample(freq).mean().dropna()
    pd.plotting.autocorrelation_plot(df.dropna())
    plt.title('Autocorrelation of ' + name)
    plt.savefig('../pics/new/2_season_ac_' + name + '.png')
    plt.show()

plot_season_ac(season_M_rep, "D", "Original", season_flag = False)

## Part 5: Integrated
df = df_train.iloc[:, 0]
data_in = df.values
data_residual = data_in[:-1] - data_in[1:]
data_trend = data_in[:-1]/data_residual

plt.plot(data_trend)
plt.show()



