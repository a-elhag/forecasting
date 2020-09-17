## ==> Part 0: Loading
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

store = pd.HDFStore('../data/power_clean.h5')

## ==> Part 1: Exploration
## Part 1a: Basic
# store['df_train'].info()
# store['df_test'].info()
# store['df_train'].hist(bins=50)

## Part 1b: Correlation
# corr_matrix = store['df_train'].corr()
# corr_matrix['Global_active_power'].sort_values(ascending=False)


## ==> Part 2: Pipelines
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

pipe_electrical = Pipeline([
    ('MinMax', MinMaxScaler())
])

pipe_date = Pipeline([
    ('Std Scaler', StandardScaler())
])

pipe_full = ColumnTransformer([
    ("elec", pipe_electrical
