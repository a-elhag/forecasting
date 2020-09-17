## ==> Part 0: Loading
import numpy as np
import pandas as pd

## Part 1: Pipelines
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
        return X.to_numpy().reshape(-1,1)

## Part 1c: Putting it all together
attribs_elec = list(store['df_train'])[:7]
attribs_date = list(store['df_train'])[7]

pipe_elec = Pipeline([
    ('MinMax', MinMaxScaler())
])

pipe_date = Pipeline([
    ('To Numpy', ToNumpy())
])

pipe_full = ColumnTransformer([
    ("elec", pipe_elec, attribs_elec),
    ("date", pipe_date, attribs_date),
],
remainder = "passthrough")

np_train = pipe_full.fit_transform(store['df_train'])
np_train.shape

## ==> Part 2
