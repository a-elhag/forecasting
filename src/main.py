## Part 0: Loading
import numpy as np
import pandas as pd

from pipeline import MyPipeline

store = pd.HDFStore('../data/power_clean.h5')

pipe = MyPipeline(store, [int(1e5), int(1e5)])
pipe.pre_fit(60)

## Next
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
reg_sgd = SGDRegressor(verbose = 1, shuffle = False)

#for split in range(pipe.max_split[0]):
for split in range(1):
    pipe.data_batch(split)
    pipe.pre_transform()
    reg_sgd.partial_fit(pipe.X, pipe.y)

rmse_final = 0

# for split in range(pipe.max_split[1]):
for split in range(1):
    pipe.data_batch(split, False)
    pipe.pre_transform()
    yhat = reg_sgd.predict(pipe.X)
    mse = mean_squared_error(pipe.y, yhat)
    rmse = np.sqrt(mse)
    rmse = pipe.reverse_minmax(rmse)

    if split != (pipe.max_split[1]-1):
        rmse_final += rmse[0][0] * pipe.range_no[1]/pipe.df_length[1]
    else:
        rmse_final += (rmse[0][0] * pipe.remain[1])/pipe.df_length[1]


print(f" SGD Test rmse = {rmse_final}")
