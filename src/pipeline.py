## ==> Part 0: Loading
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

store = pd.HDFStore('../data/power_clean.h5')

## ==> Part 1: Pipelines
## Part 1a: Imports
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

## Part 1b: Custom Pipes
class ToNumpy(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X = X.iloc[:, 0]
        year = X.dt.year.to_numpy().astype(int)
        month = X.dt.month.to_numpy().astype(int)
        day = X.dt.day.to_numpy().astype(int)
        hour = X.dt.hour.to_numpy().astype(int)
        minute = X.dt.minute.to_numpy().astype(int)
        return np.c_[year, month, day, hour, minute]

## Part 1c: Putting it all together
attribs_Y = list(store['df_train'])[0]
attribs_Y = [attribs_Y] # This is needed for 1D data
attribs_elec = list(store['df_train'])[1:7]
attribs_date = list(store['df_train'])[7]
attribs_date = [attribs_date]

pipe_elec = Pipeline([
    ('MinMax', MinMaxScaler())
])

pipe_date = Pipeline([
    ('To Numpy', ToNumpy())
])

pipe_full = ColumnTransformer([
    ("Y", pipe_elec, attribs_Y),
    ("date", pipe_date, attribs_date),
    ("elec", pipe_elec, attribs_elec),
])

np_train = pipe_full.fit_transform(store['df_train'])
## ==> Part 2
