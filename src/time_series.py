## Part 0: Loading
import keras
import numpy as np
import pandas as pd
import pipeline

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM

store = pd.HDFStore('../data/power_clean.h5')
pipe = pipeline.MyPipeline(store, [int(1e5), int(1e5)])
pipe.pre_fit(60)
pipe.data_batch(0)
pipe.pre_transform()

## Part 1: Building the LSTM
model = Sequential()
model.add(LSTM(200, activation = 'relu', input_shape=(60, 1)))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

pipe.X = pipe.X.reshape((pipe.X.shape[0], pipe.X.shape[1], 1))

model.fit(pipe.X, pipe.y, epochs = 1)

## Part 2: Testing the LSTM
def rmse(y, y_hat):
    value = 0

    for idx in range(len(y)):
        value = value + (y[idx] - y_hat[idx])**2

    value = value/len(y)
    value = value **0.5

    return value


pipe.data_batch(0, train_flag = False)
pipe.pre_transform()

pipe.X = pipe.X.reshape((pipe.X.shape[0], pipe.X.shape[1], 1))

y_hat = model.predict(pipe.X)

rmse = rmse(pipe.y, y_hat)
rmse = pipe.reverse_minmax(rmse)
print(rmse)
