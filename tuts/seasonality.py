import pandas as pd

idx = pd.Series(pd.date_range('1/1/2018', periods=100, freq='MS'),name='date')
df = pd.DataFrame(range(len(idx)), index=idx,columns=['A'])

df.loc[df.index.month.isin([1,2,3])].resample('YS').mean().values
df.loc[df.index.month.isin([1,2,3])]

df.loc[df.index.month.isin([1])].resample('MS').mean().values

for month in range(12):
    x = df.loc[df.index.month.isin([month])].resample('YS').mean()
    print(month, x.values.reshape(1,-1))

