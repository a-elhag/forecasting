## Part 0: Loading
import numpy as np
import pandas as pd
import pipeline
import matplotlib.pyplot as plt


store = pd.HDFStore('../data/power_clean.h5')
pipe = pipeline.MyPipeline(store, [int(1e5), int(1e5)])
pipe.pre_fit(60)
pipe.data_batch(0)
pipe.pre_transform()

## Autocorrelation
'''
Slow this down to an average value per hour
Then plot this
'''

df = pd.DataFrame(pipe.data[:1000, 0])

pd.plotting.autocorrelation_plot(df)
plt.show()

