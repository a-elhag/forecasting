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

## Part 0b: Sliding Window X
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

        for i in range(self.window_size):
            idx1 = i
            idx2 = row_size-2*self.window_size+i
            X_out = np.concatenate((X_out, X[idx1:idx2]), axis=1)

        return X_out[:, 1:]


train_df = BatchData(store, 'df_train', 15)
train_df.batch(0)
A = train_df.data[:, 0:2]
A.shape

trans_swx = SlidingWindowX(3)
trans_swx.fit_transform(A)

## Part 1: Split
class ToDate(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = [X[i].to_pydatetime() for i in range(len(A))]
        return np.array(X)

class SlidingWindowDate(BaseEstimator, TransformerMixin):
    def __init__(self, window_size):
        self.window_size = window_size

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[:-2*self.window_size, :]

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
    ('to date', ToDate()),
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

test_np = pipe_full.transform(store['df_test'])

test_X = test_np[:, 1:]
test_y = test_np[:, 0]
# store.close()

## Part 2: Training Models
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

reg_lin = LinearRegression(copy_X = True)
reg_lin.fit(train_X, train_y)

reg_dt = DecisionTreeRegressor(max_depth=5)
reg_dt.fit(train_X, train_y)

reg_mlp = MLPRegressor(verbose = True, batch_size = 512,
                       max_iter= 10, hidden_layer_sizes=(20, 5))
reg_mlp.fit(train_X, train_y)

# reg_rf = RandomForestRegressor(n_estimators= 10, min_samples_split= 2,
#                               min_samples_leaf= 1, verbose = True)
# reg_rf.fit(train_X, train_y)

## Part 3: Testing Models
from sklearn.metrics import mean_squared_error

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

model_error(test_X, test_y, train_X, train_y, reg_lin, 'linear')
model_error(test_X, test_y, train_X, train_y, reg_dt, 'decision tree')
model_error(test_X, test_y, train_X, train_y, reg_mlp, 'mlp')
# model_error(test_X, test_y, train_X, train_y, reg_rf)

