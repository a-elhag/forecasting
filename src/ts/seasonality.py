## Part 0: Loading
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.metrics import mean_squared_error
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.ar_model import AutoReg

store = pd.HDFStore('../../data/power_clean.h5')
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


    def seasoned(self):
        self.year = np.tile(self.year, 2)
        self.seasoned_train1 = self.df_train['2007']/self.year[:self.df_train['2007'].shape[0]]
        # self.seasoned_train1 = self.seasoned_train1 - self.df_train['2007']

        self.seasoned_train2 = self.df_train['2008']/self.year[:self.df_train['2008'].shape[0]]
        # self.seasoned_train2 = self.seasoned_train2 - self.df_train['2008']

        self.seasoned_train3 = self.df_train['2009']/self.year[:self.df_train['2009'].shape[0]]
        # self.seasoned_train3 = self.seasoned_train3 - self.df_train['2009']

        self.seasoned_train = pd.concat([self.seasoned_train1,
                                          self.seasoned_train2,
                                          self.seasoned_train3])

    def integrated(self, order):
        self.order = order
        if self.order == 0:
            self.integrated_train = self.seasoned_train.dropna()
        if self.order == 1:
            self.integrated_train = self.seasoned_train.diff().dropna()
        if self.order == 2:
            self.integrated_train = self.seasoned_train.diff().diff().dropna()

    def ar(self, lag_amount):
        self.lag_amount = lag_amount
        model_ar = AutoReg(self.integrated_train.values, lags=self.lag_amount)
        model_ar_fit = model_ar.fit()
        self.coef = model_ar_fit.params
        predictions = model_ar_fit.predict(
            start = 0, end = self.integrated_train.shape[0],
            dynamic = False)

        predict = np.zeros((self.integrated_train.shape[0]-self.lag_amount+1))

        for lag in range(1,self.lag_amount+1):
            if self.lag_amount == 1:
                self.integrated_train_lag = self.integrated_train.dropna().values
                predict = predict + self.integrated_train_lag*self.coef[lag]
                break

            if lag == 1:
                self.integrated_train_lag = self.integrated_train[self.lag_amount-1:].dropna().values
            elif lag == self.lag_amount+1:
                self.integrated_train_lag = self.integrated_train[:-self.lag_amount+1].dropna().values
            else:
                idx1 = -lag + self.lag_amount
                idx2 = -lag + 1
                self.integrated_train_lag = self.integrated_train[idx1:idx2].dropna().values

            predict = predict + self.integrated_train_lag*self.coef[lag]

        predict = predict + self.coef[0]
        error = (predict-predictions).sum()

        if self.lag_amount==1:
            self.rmse = np.sqrt(mean_squared_error(predict, self.integrated_train.values))
            self.ar_train = self.integrated_train - predict
        elif self.lag_amount==0:
            self.rmse = np.sqrt(mean_squared_error(predict[:-1], self.integrated_train.values))
            self.ar_train = self.integrated_train - predict[:-1]
        else:
            idx1 = (self.lag_amount-1)*2
            idx2 = -(self.lag_amount-1)

            self.rmse = np.sqrt(mean_squared_error(predict[:idx2], self.integrated_train.values[idx1:]))
            self.ar_train = self.integrated_train[idx1:] - predict[:idx2]


    def ma(self, lag_ma_amount):
        self.lag_ma_amount = lag_ma_amount
        model_ma = AutoReg(self.ar_train.values, lags=self.lag_ma_amount)
        model_ma_fit = model_ma.fit()
        self.coef_ma = model_ma_fit.params
        predictions = model_ma_fit.predict(
            start = 0, end = self.ar_train.shape[0],
            dynamic = False)

        predict = np.zeros((self.ar_train.shape[0]-lag_ma_amount+1))

        for lag in range(1,self.lag_ma_amount+1):
            if self.lag_ma_amount == 1:
                self.ar_train_lag = self.ar_train.dropna().values
                predict = predict + self.ar_train_lag*self.coef_ma[lag]
                break

            if lag == 1:
                self.ar_train_lag = self.ar_train[self.lag_ma_amount-1:].dropna().values
            elif lag == self.lag_ma_amount+1:
                self.ar_train_lag = self.ar_train[:-self.lag_ma_amount+1].dropna().values
            else:
                idx1 = -lag + self.lag_ma_amount
                idx2 = -lag + 1
                self.ar_train_lag = self.ar_train[idx1:idx2].dropna().values

            predict = predict + self.ar_train_lag*self.coef_ma[lag]

        predict = predict + self.coef_ma[0]
        error = (predict-predictions).sum()

        if self.lag_ma_amount==1:
            self.rmse = np.sqrt(mean_squared_error(predict, self.ar_train.values))
            self.ma_train = self.ar_train - predict
        elif self.lag_ma_amount==0:
            self.rmse = np.sqrt(mean_squared_error(predict[:-1], self.ar_train.values))
            self.ma_train = self.ar_train - predict[:-1]
        else:
            idx1 = (self.lag_ma_amount-1)*2
            idx2 = -(self.lag_ma_amount-1)

            self.rmse = np.sqrt(mean_squared_error(predict[:idx2], self.ar_train.values[idx1:]))
            self.ma_train = self.ar_train[idx1:] - predict[:idx2]

    def test_ac(self):
        '''
        Calculate OLS then find durbin_watson for residuals
        0 < ans < 2 ==> Positive correlation
        2 < ans < 4 ==> Negative correlation
        ans == 2    ==> Zero correlation

        ols_res = OLS(np.random.rand(1000), np.ones(1000)).fit()
        durbin_watson(ols_res.resid)
        '''

        resampled = self.ma_train
        ols_res = OLS(resampled, np.ones(
            resampled.shape[0])).fit()
        self.test_durbin_watson = durbin_watson(ols_res.resid)

    def seasonality_search(self, order_season):
        
        self.order_season = order_season

        if self.order_season == 0:
            self.tag_season = "Nothing"
            self.seasoned()
        elif self.order_season == 1:
            self.tag_season = "D:365"
            self.get_all("D", 365)
            self.seasoned()
        elif self.order_season == 2:
            self.tag_season = "H:24"
            self.get_all("H", 24)
            self.seasoned()
        elif self.order_season == 3:
            self.tag_season = "H:24*7"
            self.get_all("H", 24*7)
            self.seasoned()
        elif self.order_season == 4:
            self.tag_season = "D:365 + H:24"
            self.get_all("D", 365)
            self.get_all("H", 24)
            self.seasoned()
        elif self.order_season == 5:
            self.tag_season = "D:365 + H:24*7"
            self.get_all("D", 365)
            self.get_all("H", 24*7)
            self.seasoned()
        elif self.order_season == 6:
            self.tag_season = "D:365 + H:24 + H:24*7"
            self.get_all("D", 365)
            self.get_all("H", 24)
            self.get_all("H", 24*7)
            self.seasoned()

    def normal_search(self):
        pass

    def grid_search(self):
        self.values = []
        self.keys = []
        for sea in range(7):
            self.seasonality_search(sea)
            for i in range(3):
                self.integrated(i)
                for lag in range(3):
                    self.ar(lag)
                    for lag_ma in range(3):
                        self.ma(lag_ma)

                        self.test_ac()
                        string=f"{self.tag_season}, i: {i}, ar: {lag}, ma: {lag_ma}"
                        print(string)
                        print(self.test_durbin_watson)
                        print("--------------")
                        self.values.append(self.test_durbin_watson)
                        self.keys.append(string)

        self.results_df = pd.DataFrame(
            {'keys': self.keys,
             'values': self.values
            })

        self.results_df.iloc[:,1] = self.results_df.iloc[:,1] - 2
        self.results_df.iloc[:,1] = np.abs(self.results_df.iloc[:,1])
        self.idx_min = self.results_df.iloc[:,1].idxmin()

    def plot_year(self):
        plt.plot(self.year[::60])
        plt.show()

    def plot_ac(self):
        pd.plotting.autocorrelation_plot(
            self.ma_train.resample("H").mean().dropna())
        plt.show()


## Part 2: After class
season = Season(df_train, df_test)
season.grid_search()
season.results_df.iloc[season.idx_min]

