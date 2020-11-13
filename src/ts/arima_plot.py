'''
Go to seasonality.py to pick up
this
'''
## Part 2: AR
from statsmodels.graphics.tsaplots import plot_pacf

plot_pacf(season.ar_train, lags=10) # 1 is significant
plt.show()

## Part 3: MA
from statsmodels.graphics.tsaplots import plot_acf

plot_acf(season.integrated_train.resample("D").mean().dropna(), lags=10) # 1 is significant
plt.show()

## Part 4: ARIMA
from statsmodels.tsa.arima_model import ARIMA

data_in = season.df_train.resample("D").mean().dropna().values
model = ARIMA(data_in, order=(1,1,1))
model_fit = model.fit(disp=0)
print(model_fit.summary())


