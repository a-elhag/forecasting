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
class Season():
    '''
    This is a class to remove the seasonality from the input data
    set
    '''
    def __init__(self, df_train, df_test):
        self.df_train = df_train.iloc[:,0]
        self.df_test = df_test.iloc[:,0]


    def resample(self, freq, flag_train=True):
        ''' Resample the dataset
        MS ==> Month (Day 1)
        W ==> Week
        D ==> Day of Year
        H ==> Hour
        '''
        self.freq = freq
        self.flag_train = flag_train

        if self.flag_train:
            self.rs_train = self.df_train.resample(self.freq).mean()
        else:
            self.rs_test = self.df_test.resample(self.freq).mean()

    def get_idx(self):
        '''
        Have to subtract 1 from most of the indeces because
        they are not zero indexed.
        Idiots
        '''

        if self.flag_train:
            self.idx_train = {
                "MS": self.rs_train.index.month - 1,
                "W": pd.Int64Index(self.rs_train.index.isocalendar().week) - 1,
                "D": self.rs_train.index.dayofyear - 1,
                "H": self.rs_train.index.hour}
        else:
            self.idx_test = {
                "MS": self.rs_test.index.month - 1,
                "W": pd.Int64Index(self.rs_test.index.isocalendar().week) - 1,
                "D": self.rs_test.index.dayofyear - 1,
                "H": self.rs_test.index.hour}

    def get_pattern(self, period):
        self.period = period

        if self.flag_train:
            self.season = np.zeros(self.period)
            idx = np.arange(self.idx_train[self.freq].values.shape[0])

            for p in range(self.period):
                idx_temp = idx[p::self.period]
                self.season[p] = np.nanmean(
                    self.rs_train.iloc[idx_temp].values)
        else:
            print(f"Min {self.freq} :", self.idx_test[self.freq].min())
            print(f"Max {self.freq} :", self.idx_test[self.freq].max())


season = Season(df_train, df_test)
season.resample("H")
season.get_idx()
season.get_pattern(24*7)
season.season.shape
plt.plot(season.season)
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

plot_season_ac(season_M_rep, "H", "Original in Hours", season_flag = False)
plot_season_ac(season_M_rep, "D", "Original in Days", season_flag = False)
plot_season_ac(season_M_rep, "H", "Residual of Months in Hours")
plot_season_ac(season_M_rep, "D", "Residual of Months in Days")
plot_season_ac(season_W_rep, "H", "Residual of Weeks in Hours")
plot_season_ac(season_W_rep, "D", "Residual of Weeks in Days")
plot_season_ac(season_D_rep, "H", "Residual of Days in Hours")
plot_season_ac(season_D_rep, "D", "Residual of Days in Days")

plot_season_ac(season_H_rep, "H", "Residual of Hours in Hours")
plot_season_ac(season_D_rep*season_H_rep, "H", "Combo Residual of Days+Hours in Hours")

## Part 5: Integrated
df = df_train.iloc[:, 0]
data_in = df.values
data_residual = data_in[:-1] - data_in[1:]

fig, axs = plt.subplots(2)
axs[0].plot(data_in)
axs[0].set_title('Original')
axs[1].plot(data_residual)
axs[1].set_title('Residual')

plt.tight_layout()
plt.grid()
plt.savefig('../pics/new/3_integrated.png')
plt.show()


