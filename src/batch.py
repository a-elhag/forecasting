## Part 0: Loading
import numpy as np
import pandas as pd

store = pd.HDFStore('../data/power_clean.h5')
train_iter = store['df_train'].iterrows()

A = np.array([])
for _ in range(100):
    next(train_iter)

A = next(train_iter)[1].to_numpy()
A = np.vstack((A, next(train_iter)[1].to_numpy()))
A
## Part 1: Iterator


## Part 1: Pipes
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

class SplitDate(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
#         store['df_train']['DateTime'].dt.to_pydatetime()
        X = X.iloc[:, 0]

        weekday =  X.dt.dayofweek.to_numpy()
        weekday[weekday < 5] = 1 # weekday=1, weekend=0
        weekday[weekday >=5] = 0

        year = X.dt.year.to_numpy().astype(int)
        month = X.dt.month.to_numpy().astype(int)
        day = X.dt.day.to_numpy().astype(int)
        hour = X.dt.hour.to_numpy().astype(int)
        minute = X.dt.minute.to_numpy().astype(int)
        return np.c_[year, month, day, hour, minute]

class SlidingWindowDate(BaseEstimator, TransformerMixin):
    def __init__(self, window_size):
        self.window_size = window_size

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[:-2*self.window_size, :]

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
        row_size = X.shape[0]
        X_out = np.zeros((row_size-2*self.window_size, 1))

        for i in range(self.window_size):
            idx1 = i
            idx2 = row_size-2*self.window_size+i
            X_out = np.concatenate((X_out, X[idx1:idx2]), axis=1)

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
        if not type(X).__module__ == 'numpy':
            X = X.iloc[:, 0].to_numpy()
        X = X.reshape(-1, 1)
        return X[window_size*2:, :]

attribs_Y = list(store['df_train'])[0]
attribs_Y = [attribs_Y] # This is needed for 1D data
attribs_elec = list(store['df_train'])[0:3]
attribs_date = list(store['df_train'])[7]
attribs_date = [attribs_date]

window_size = 10
pipe_Y = Pipeline([
    ('min-max', MinMaxScaler()),
    ('window', SlidingWindowY(window_size))
])

pipe_elec = Pipeline([
    ('min-max', MinMaxScaler()),
    ('window', SlidingWindowX(window_size))
])

pipe_date = Pipeline([
    ('split date', SplitDate()),
    ('window', SlidingWindowDate(window_size))
])

pipe_full = ColumnTransformer([
    ("Y", pipe_Y, attribs_Y),
    ("elec", pipe_elec, attribs_elec),
    ("date", pipe_date, attribs_date),
])

train_np = pipe_full.fit_transform(store['df_train'])

train_X = train_np[:, 1:]
train_y = train_np[:, 0]
