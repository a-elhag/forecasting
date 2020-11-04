## Part 1: Imports
import numpy as np
import pandas as pd
from src.seasonality import Season

store = pd.HDFStore('data/power_clean.h5')
df_train = store['df_train']
df_test = store['df_test']
store.close()

df_train.set_index('DateTime', inplace=True)
df_test.set_index('DateTime', inplace=True)

## Part 2: Seasonality
season = Season(df_train, df_test)

def test_idx():
    assert season.idx_train["MS"].min() == 0
    assert season.idx_train["W"].min() == 0
    assert season.idx_train["D"].min() == 0
    assert season.idx_train["H"].min() == 0
    assert season.idx_train["MS"].max() == 11
    assert season.idx_train["W"].max() == (52 or 53)
    assert season.idx_train["D"].max() == (365 or 366)
    assert season.idx_train["H"].max() == 23

    assert season.idx_test["MS"].min() == 0
    assert season.idx_test["W"].min() == 0
    assert season.idx_test["D"].min() == 0
    assert season.idx_test["H"].min() == 0
    assert season.idx_test["MS"].max() == 10
    assert season.idx_test["W"].max() ==  52
    assert season.idx_test["D"].max() == 329
    assert season.idx_test["H"].max() == 23
