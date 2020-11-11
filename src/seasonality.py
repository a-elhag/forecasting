## Part 0: Loading
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

# Stats Models
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.stattools import durbin_watson

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


    def transform(self):
        self.year = np.tile(self.year, 2)
        self.transform_train1 = self.df_train['2007']/self.year[:self.df_train['2007'].shape[0]]
        self.transform_train2 = self.df_train['2008']/self.year[:self.df_train['2008'].shape[0]]
        self.transform_train3 = self.df_train['2009']/self.year[:self.df_train['2009'].shape[0]]

        self.transform_train = pd.concat([self.transform_train1,
                                          self.transform_train2,
                                          self.transform_train3])
        self.residuals_train = self.df_train - self.transform_train

    def test_ac(self):
        '''
        Calculate OLS then find durbin_watson for residuals
        0 < ans < 2 ==> Positive correlation
        2 < ans < 4 ==> Negative correlation
        ans == 2    ==> Zero correlation

        ols_res = OLS(np.random.rand(1000), np.ones(1000)).fit()
        durbin_watson(ols_res.resid)
        '''

        ols_res = OLS(self.transform_train, np.ones(
            self.transform_train.shape[0])).fit()
        return durbin_watson(ols_res.resid)

    def plot_year(self):
        plt.plot(self.year[::60])
        plt.show()

    def plot_ac(self):
        pd.plotting.autocorrelation_plot(
            self.transform_train.resample("D").mean().dropna())
        plt.show()


## Part 1: After class

season = Season(df_train, df_test)
season.get_all("D", 365)
season.get_all("H", 24)
season.transform()
season.test_ac()


