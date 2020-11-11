## Part 0: Loading
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

store = pd.HDFStore('../data/power_clean.h5')
df_train = store['df_train']
df_test = store['df_test']
store.close()
df_train.set_index('DateTime', inplace=True)
df_test.set_index('DateTime', inplace=True)

idx = df_train.index >= '2007'
df_train = df_train[idx]
## Part 1: Formatting Data
class Season():
    '''
    This is a class to remove the seasonality from the input data
    set
    '''
    def __init__(self, df_train, df_test):
        self.df_train = df_train.iloc[:,0]
        self.df_test = df_test.iloc[:,0]

        self.season_train = np.ones(df_train.size)
        self.season_test = np.ones(df_test.size)

        self.year = np.ones(60*8760)

    def resample(self, freq):
        ''' Resample the dataset
        MS ==> Month (Day 1)
        W ==> Week
        D ==> Day of Year
        H ==> Hour
        '''
        self.freq = freq
        self.rs_train = self.df_train.resample(self.freq).mean()

    def get_idx(self):
        '''
        Have to subtract 1 from most of the indeces because
        they are not zero indexed.
        Idiots
        '''

        self.idx_train = {
            "MS": self.rs_train.index.month - 1,
            "W": pd.Int64Index(self.rs_train.index.isocalendar().week) - 1,
            "D": self.rs_train.index.dayofyear - 1,
            "H": self.rs_train.index.hour}

    def get_pattern(self, period):
        self.period = period
        self.season = np.zeros(self.period)

        idx_zero = np.where((self.idx_train[self.freq].values==0))[0][0]
        idx = np.arange(self.idx_train[self.freq].values.shape[0] - idx_zero) + idx_zero

        for p in range(self.period):
            idx_temp = idx[p::self.period]
            self.season[p] = np.nanmean(
                self.rs_train.iloc[idx_temp].values)


    def get_year(self):
        # rate = 2
        # length = self.season.shape[0]
        # self.year_temp = signal.resample(self.season, rate*length)
        # self.year_temp = signal.resample_poly(self.season, rate, 1)

        if self.freq == "D":
            time_repeat = int(np.ceil(60*8760/self.period))
            self.year_temp = np.repeat(self.season, time_repeat)

        if self.freq == "H":
            time_repeat = 60
            self.year_temp = np.repeat(self.season, time_repeat)
            time_tile = int(np.ceil(8760/self.period))
            self.year_temp = np.tile(self.year_temp, time_tile)

        self.year_temp = self.year_temp[:60*8760]
        self.year = self.year[:60*8760]*self.year_temp

    def get_all(self, freq, period):
        self.resample(freq)
        self.get_idx()
        self.get_pattern(period)
        self.get_year()

    def plot_year(self):
        plt.plot(self.year[::60])
        plt.show()

    def transform(self):
        self.year = np.tile(self.year, 2)
        self.transform_train1 = self.df_train['2007']/self.year[:self.df_train['2007'].shape[0]]
        self.transform_train2 = self.df_train['2008']/self.year[:self.df_train['2008'].shape[0]]
        self.transform_train3 = self.df_train['2009']/self.year[:self.df_train['2009'].shape[0]]

        self.transform_train = pd.concat([self.transform_train1,
                                          self.transform_train2,
                                          self.transform_train3])




## Part 1: After class
season = Season(df_train, df_test)
season.get_all("D", 365)
season.get_all("H", 24*7)
season.transform()

plt.plot(season.transform_train)
plt.show()


## Part 4: Autocorrelation
from statsmodels.graphics.tsaplots import plot_acf
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
