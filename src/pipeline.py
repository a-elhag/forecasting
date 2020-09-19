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

class SlidingWindow(BaseEstimator, TransformerMixin):
    def __init__(self, size):
        self.size = size

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        pass



attribs_Y = list(store['df_train'])[0]
attribs_Y = [attribs_Y] # This is needed for 1D data
attribs_elec = list(store['df_train'])[1:7]
attribs_date = list(store['df_train'])[7]
attribs_date = [attribs_date]

pipe_elec = Pipeline([
    ('min-max', MinMaxScaler())
])

pipe_date = Pipeline([
    ('split date', SplitDate())
])

pipe_full = ColumnTransformer([
    ("Y", pipe_elec, attribs_Y),
    ("elec", pipe_elec, attribs_elec),
    ("date", pipe_date, attribs_date),
])

train_np = pipe_full.fit_transform(store['df_train'])
test_np = pipe_full.transform(store['df_test'])

train_X = train_np[:, 1:]
train_y = train_np[:, 0]

test_X = test_np[:, 1:]
test_y = test_np[:, 0]

# store.close()
## ==> Part2: Models
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

reg_lin = LinearRegression()
reg_lin.fit(train_X, train_y)

test_y_predict = reg_lin.predict(test_X)

mse_lin = mean_squared_error(test_y, test_y_predict)
mse_lin = np.sqrt(mse_lin)

pipe_full.named_transformers_['Y'].inverse_transform([[mse_lin]])

## Part 3: Testing
def windows(X, window_size):
    row_output = len(X) - window_size + 1 # Go through all values, except at the very end we only can ge t 1
    examples = []

    for i in range(row_output):
        example = X[i:i+window_size]
        examples.append(np.expand_dims(example, 0))

    return np.vstack(examples)


c = ['A','B','C','D','E', 'F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

windows(c, 4)
