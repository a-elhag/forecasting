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

## Part 1: Creating Transformers!
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
        return np.array(X)

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

## Part 3: Making pipelines!
attribs_Y = np.array([0])
attribs_elec = np.arange(0, 7)
attribs_date = np.array([7])

window_size = 3
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
    ('window', SlidingWindowX(window_size))
])

pipe_full = ColumnTransformer([
    ("Y", pipe_Y, attribs_Y),
    ("elec", pipe_elec, attribs_elec),
    ("date", pipe_date, attribs_date),
])

train_batch = BatchData(store['df_train'], 500000)
train_batch.full()
pipe_full.fit(train_batch.data)

## Part 4: Applying Pipelines
train_batch.batch(0)
train_np = pipe_full.transform(train_batch.data)

train_X = train_np[:, 1:7]
train_y = train_np[:, 0]


test_batch = BatchData(store['df_test'], 500000)
test_batch.batch(0)
test_np = pipe_full.transform(test_batch.data)

test_X = test_np[:, 1:7]
test_y = test_np[:, 0]
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

