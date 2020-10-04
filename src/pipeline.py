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
        if train_flag:
            idx1 = split*self.range_no[0]
            idx2 = (split+1)*self.range_no[0]
            self.data = self.store[0].iloc[idx1:idx2, :].to_numpy()
        else:
            idx1 = split*self.range_no[1]
            idx2 = (split+1)*self.range_no[1]
            self.data = self.store[1].iloc[idx1:idx2, :].to_numpy()

    def data_full(self, train_flag = True):
        if train_flag:
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


pipe = MyPipeline(store, [int(2e5), int(1e5)])
pipe.data_batch(0, False)
pipe.data.shape



## Part 2: Something
train_batch = BatchData(store['df_train'], 200000)
train_batch.full()
pipe_full.fit(train_batch.data)


## Part 3: Simple Reset
def reset_window_size(pipe, window_size):
    """
    Need to do this because sklearn is acting like a *****, will always call 
    fit_transform when you call fit. Thus we set window_size=1 initially and then
    we go CRAZY after it
    """
    pipe.set_params(Y__window__window_size = window_size)
    pipe.set_params(elec__window__window_size = window_size)
#     pipe.set_params(date__window__window_size = window_size)
    pipe.named_transformers_['Y']['window'].window_size = window_size
    pipe.named_transformers_['elec']['window'].window_size = window_size
#     pipe.named_transformers_['date']['window'].window_size = window_size
    return pipe

window_size = 60
pipe_full = reset_window_size(pipe_full, window_size)

## Part 4: Applying Pipelines
train_batch.batch(1)
train_np = pipe_full.transform(train_batch.data)
train_batch.data.shape

train_X = train_np[:, 1:]
train_y = train_np[:, 0]

## Part 5: Batch Training Models
from sklearn.linear_model import SGDRegressor

reg_sgd = SGDRegressor(verbose = 1, shuffle = False)

for split in range(train_batch.max_split):
    print(f"Split: {split} out of {train_batch.max_split}")
    train_batch.batch(split)
    train_np = pipe_full.transform(train_batch.data)

    train_X = train_np[:, 1:]
    train_y = train_np[:, 0]

    for _ in range(5):
        reg_sgd.partial_fit(train_X, train_y)

## Part 6: Testing Models
from sklearn.metrics import mean_squared_error

test_batch = BatchData(store['df_test'], 100000)
test_batch.df_length
test_batch.range_no
test_batch.max_split
remain = test_batch.df_length - (
    (test_batch.max_split - 1) * test_batch.range_no)

rmse_final = 0
for split in range(test_batch.max_split):
    print(f"Split: {split}")
    test_batch.batch(split)
    test_np = pipe_full.transform(test_batch.data)

    test_X = test_np[:, 1:]
    test_y = test_np[:, 0]

    test_yhat = reg_sgd.predict(test_X)
    mse = mean_squared_error(test_y, test_yhat)
    rmse = np.sqrt(mse)
    rmse = pipe_full.named_transformers_['Y']['min-max'].inverse_transform([[rmse]])
    if split == (test_batch.max_split-1):
        rmse_final += (rmse[0][0] * remain)/test_batch.df_length
    else:
        rmse_final += rmse[0][0] * test_batch.range_no/test_batch.df_length

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


