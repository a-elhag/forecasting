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

## Part 2: Making pipelines!
attribs_Y = np.array([0])
attribs_elec = np.arange(0, 7)
attribs_date = np.array([7])

pipe_Y = Pipeline([
    ('min-max', MinMaxScaler()),
    ('window', SlidingWindowY(2))
])

pipe_elec = Pipeline([
    ('min-max', MinMaxScaler()),
    ('window', SlidingWindowX(2))
])

pipe_date = Pipeline([
    ('to date', ToDate()),
    ('window', SlidingWindowX(2))
])

pipe_full = ColumnTransformer([
    ("Y", pipe_Y, attribs_Y),
    ("elec", pipe_elec, attribs_elec),
#     ("date", pipe_date, attribs_date),
])

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

# del train_np, train_X, train_y

## Part 5: Batch Training Models
from sklearn.linear_model import SGDRegressor

reg_sgd = SGDRegressor(verbose = 1, shuffle = False)

for split in range(train_batch.max_split):
    print(f"Split: {split} out of {train_batch.max_split}")
    train_batch.batch(split)
    train_np = pipe_full.transform(train_batch.data)

    train_X = train_np[:, 1:]
    train_y = train_np[:, 0]

    for _ in range(3):
        reg_sgd.partial_fit(train_X, train_y)

## Part 5: Training Models
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

test_batch = BatchData(store['df_test'], 100000)
test_batch.batch(0)
test_np = pipe_full.transform(test_batch.data)

test_X = test_np[:, 1:]
test_y = test_np[:, 0]

test_yhat = reg_sgd.predict(test_X)
mse = mean_squared_error(test_y, test_yhat)
rmse = np.sqrt(mse)
rmse = pipe_full.named_transformers_['Y']['min-max'].inverse_transform([[rmse]])
print(f" SGD Test rmse = {rmse}")


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

model_error(test_X, test_y, train_X, train_y, reg_lin, 'linear')
model_error(test_X, test_y, train_X, train_y, reg_dt, 'decision tree')
model_error(test_X, test_y, train_X, train_y, reg_mlp, 'MLP')
# model_error(test_X, test_y, train_X, train_y, reg_rf)

