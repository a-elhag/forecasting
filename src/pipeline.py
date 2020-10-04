## Part 0: Loading
from batch import BatchData
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

store = pd.HDFStore('../data/power_clean.h5')

class ToDate(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if type(X[0]) == "pandas._libs.tslibs.timestamps.Timestamp":
            X = [X[i].to_pydatetime() for i in range(len(X))]
        else:
            X = [X[i][0].to_pydatetime() for i in range(len(X))]
        return np.array(X).reshape(-1, 1)

class SlidingWindowX(BaseEstimator, TransformerMixin):
    def __init__(self, window_size):
        self.window_size = window_size

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        '''
        Creates a sliding window over an input that has the shape of
        (rows, features) for X
        '''

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        row_size = X.shape[0]
        X_out = np.zeros((row_size-2*self.window_size, 1))

        for j in range(X.shape[1]):
            for i in range(self.window_size):
                idx1 = i
                idx2 = row_size-2*self.window_size+i
                X_out = np.concatenate((X_out, X[idx1:idx2, j].reshape(-1, 1)), axis=1)

        return X_out[:, 1:]

class SlidingWindowY(BaseEstimator, TransformerMixin):
    def __init__(self, window_size):
        self.window_size = window_size

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        '''
        Creates a sliding window over an input that has the shape of
        (rows, features) for Y
        '''
        X = X.reshape(-1, 1)
        return X[self.window_size*2:, :]

class MyPipeline():
    def __init__(self, store, range_no):
        self.store = [store['df_train'], store['df_test']]
        self.df_length = [len(self.store[0]), len(self.store[1])]
        self.range_no = range_no

        self.max_split = [0, 0]
        self.remain = [0, 0]
        for _ in range(2):
            self.max_split[_] = (self.df_length[_]//self.range_no[_] + 1)
            self.remain[_] = self.df_length[_] - (
                (self.max_split[_] - 1) * self.range_no[_])

        self.pipe_setting()


    def data_batch(self, split, train_flag = True):
        self.split = split
        self.train_flag = train_flag
        self.data_trans = self.X = self.y = 0
        if self.train_flag:
            idx1 = self.split*self.range_no[0]
            idx2 = (self.split+1)*self.range_no[0]
            self.data = self.store[0].iloc[idx1:idx2, :].to_numpy()

        else:
            idx1 = self.split*self.range_no[1]
            idx2 = (self.split+1)*self.range_no[1]
            self.data = self.store[1].iloc[idx1:idx2, :].to_numpy()


    def data_full(self, train_flag = True):
        self.train_flag = train_flag
        self.data_trans = self.X = self.y = 0
        if self.train_flag:
            self.data = self.store[0].iloc[:, :].to_numpy()
        else:
            self.data = self.store[1].iloc[:, :].to_numpy()

    def data_names(self):
        return list(self.store[0])

    def pipe_setting(self):
        self.attribs_Y = np.array([0])
        self.attribs_elec = np.arange(0, 7)
        self.attribs_date = np.array([7])

        self.pipe_Y = Pipeline([
            ('min-max', MinMaxScaler()),
            ('window', SlidingWindowY(1))
        ])

        self.pipe_elec = Pipeline([
            ('min-max', MinMaxScaler()),
            ('window', SlidingWindowX(1))
        ])

        self.pipe_date = Pipeline([
            ('to date', ToDate()),
            ('window', SlidingWindowX(1))
        ])

        self.pipe_full = ColumnTransformer([
            ("Y", self.pipe_Y, self.attribs_Y),
            ("elec", self.pipe_elec, self.attribs_elec),
        #     ("date", self.pipe_date, self.attribs_date),
        ])

    def pre_fit(self, window_size, verbose = True):
        if verbose:
            print("Fitting Data")


        self.data_full()
        self.pipe_full.fit(self.data)

        """
        Need to do this because sklearn is acting like a *****, will always call 
        fit_transform when you call fit. Thus we set window_size=1 initially and then
        we go CRAZY after it
        """
        self.pipe_full.set_params(Y__window__window_size = window_size)
        self.pipe_full.set_params(elec__window__window_size = window_size)
      # self.pipe_full.set_params(date__window__window_size = window_size)
        self.pipe_full.named_transformers_['Y']['window'].window_size = window_size
        self.pipe_full.named_transformers_['elec']['window'].window_size = window_size
      # self.pipe_full.named_transformers_['date']['window'].window_size = window_size

    def pre_transform(self, verbose = True):
        if verbose:
            if self.train_flag:
                print(f"Transform Train: {self.split} out of {self.max_split[0]}")
            else:
                print(f"Transform Test: {self.split} out of {self.max_split[1]}")

        self.data_trans = self.pipe_full.transform(self.data)
        self.X = self.data_trans[:, 1:]
        self.y = self.data_trans[:, 0]

    def reverse_minmax(self, value):
        if not isinstance(value, np.ndarray):
            value = np.array([[value]])

        if value.ndim == 1:
            value = value.reshape(1, -1)
        
        output = self.pipe_full.named_transformers_['Y']['min-max'].inverse_transform(value)
        return output

pipe = MyPipeline(store, [int(1e5), int(1e5)])
pipe.pre_fit(60)

## Next
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
reg_sgd = SGDRegressor(verbose = 1, shuffle = False)

for split in range(pipe.max_split[0]):
    pipe.data_batch(split)
    pipe.pre_transform()
    reg_sgd.partial_fit(pipe.X, pipe.y)

split = 0
rmse_final = 0

for split in range(pipe.max_split[1]):
    pipe.data_batch(split, False)
    pipe.pre_transform()
    yhat = reg_sgd.predict(pipe.X)
    mse = mean_squared_error(pipe.y, yhat)
    rmse = np.sqrt(mse)
    rmse = pipe.pipe_full.named_transformers_['Y']['min-max'].inverse_transform([[rmse]])

    if split != (pipe.max_split[1]-1):
        rmse_final += rmse[0][0] * pipe.range_no[1]/pipe.df_length[1]
    else:
        rmse_final += (rmse[0][0] * pipe.remain[1])/pipe.df_length[1]


print(f" SGD Test rmse = {rmse_final}")

## Part Else
def model_error(test_X, test_y, train_X, train_y, reg, name):
    train_yhat = reg.predict(train_X)
    mse = mean_squared_error(train_y, train_yhat)
    rmse = np.sqrt(mse)
    rmse = pipe_full.named_transformers_['Y']['min-max'].inverse_transform([[rmse]])
    print(f"{name} Train rmse = {rmse}")
    test_yhat = reg.predict(test_X)
    mse = mean_squared_error(test_y, test_yhat)
    rmse = np.sqrt(mse)
    rmse = pipe_full.named_transformers_['Y']['min-max'].inverse_transform([[rmse]])
    print(f"{name} Test rmse = {rmse}")


