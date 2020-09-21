## Part 0: Loading
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

store = pd.HDFStore('../data/power_clean.h5')

## Part 1: Pipes
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
        row_size, col_size = X.shape
        X_out = np.zeros((row_size-self.window_size, 1))

        for i in range(self.window_size):
            X_out = np.concatenate((X_out, X[i:row_size-self.window_size+i]), axis=1)

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
        return X[window_size:, :]

attribs_Y = list(store['df_train'])[0]
attribs_Y = [attribs_Y] # This is needed for 1D data
attribs_elec = list(store['df_train'])[1:7]
attribs_date = list(store['df_train'])[7]
attribs_date = [attribs_date]

window_size = 20
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
    ('window', SlidingWindowX(window_size))
])

pipe_full = ColumnTransformer([
    ("Y", pipe_Y, attribs_Y),
    ("elec", pipe_elec, attribs_elec),
    ("date", pipe_date, attribs_date),
])

train_np = pipe_full.fit_transform(store['df_train'])

train_X = train_np[:, 1:]
train_y = train_np[:, 0]

# store.close()
## ==> Part2: Training Models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

reg_lin = LinearRegression(copy_X = True)
reg_lin.fit(train_X, train_y)

clf_dt = DecisionTreeRegressor(max_depth=1)
clf_dt.fit(train_X, train_y)

## Part 3: Testing Models
from sklearn.metrics import mean_squared_error
test_np = pipe_full.transform(store['df_test'])

test_X = test_np[:, 1:]
test_y = test_np[:, 0]

def predict(clf, X, y):

test_yhat = reg_lin.predict(test_X)

mse_lin = mean_squared_error(test_y, test_yhat)
mse_lin = np.sqrt(mse_lin)
pipe_full.named_transformers_['Y']['min-max'].inverse_transform([[mse_lin]])

##
