import pandas as pd

class BatchData():
    def __init__(self, store, range_no):
        self.store = store
        self.df_length = len(self.store)
        self.range_no = range_no
        self.max_split = (self.df_length//self.range_no + 1)

        self.remain = self.df_length - ((self.max_split - 1) * self.range_no)


    def batch(self, split):
        idx1 = split*self.range_no
        idx2 = (split+1)*self.range_no
        self.data = self.store.iloc[idx1:idx2, :].to_numpy()

    def full(self):
        self.data = self.store.iloc[:, :].to_numpy()

    def names(self):
        self.names = list(self.store)

if __name__ == '__main__':
    store = pd.HDFStore('../data/power_clean.h5')
    train_batch = BatchData(store, 'df_train', 500000)
    train_batch.batch(0)

    test_batch = BatchData(store, 'df_test', 500000)
    test_batch.batch(0)
