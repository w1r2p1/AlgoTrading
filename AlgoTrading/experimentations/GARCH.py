import numpy as np
import pyflux as pf
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

jpm = DataReader('JPM',  'yahoo', datetime(2006,1,1), datetime(2016,3,10))
returns = pd.DataFrame(np.diff(np.log(jpm['Adj Close'].values)))
returns.index = jpm.index.values[1:jpm.index.values.shape[0]]
returns.columns = ['JPM Returns']

# Get the dataframe from the csv file
quote           = 'BTC'
pair            = 'ADABTC'
timeframe       = '1h'
ClosesOrReturns = ''



print("_________ " + pair + " _________")
filename = 'Historical_data/' + quote + '/' + pair + '_' + timeframe
df = pd.read_csv(filename, sep='\t')[1:200]
returns = np.log1p(df.close.pct_change())


am = arch_model(returns, rescale=True)
start_loc = 0
end_loc   = 63            # 3 month min rolling windows
forecasts = {}
print(len(returns) - end_loc)

for i in range(1, len(returns) - end_loc):
    # print(i)
    fitted = am.fit(first_obs=i, last_obs=i+end_loc, disp='off')
    temp   = np.sqrt(fitted.forecast(horizon=1).variance)
    fcast  = temp.iloc[i + end_loc - 1]
    forecasts[fcast.name] = fcast

# for key, value in forecasts.items():
#     print(key, value)


# This will give us an annualized volatility forecast for each day. We shift because arch_model gives you the forecast for the next day, by default. We want the data to be associated on the day that the forecast is for.
arch_vol     = np.sqrt(252)*(pd.DataFrame(forecasts).T / am.scale).shift(1).iloc[1:]['h.1']
# If your goal is to calculate volatility from returns, first you need to decide on a time period for the rolling window calculating. I'm going to use one month:
realized_vol = np.sqrt(252)*df.close.rolling(21).std().iloc[end_loc:]

fig, ax1 = plt.subplots()
ax1.plot(df.close, color='g')
ax2 = ax1.twinx()
ax2.plot(realized_vol, color='r')

# plt.plot(realized_vol)
# plt.plot(df.close)
plt.show()

# mape = np.mean(np.abs((realized_vol - arch_vol) / realized_vol))
# corr = arch_vol.corr(realized_vol)



# rolling_predictions = []
# test_size = 1000
#
# for i in range(test_size):
#     train = returns[:-(test_size-i)]
#     model = arch_model(train, p=1, q=1)
#     model_fit = model.fit(disp='off')
#     pred = model_fit.forecast(horizon=1)
#     rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))
#
# # The result of the forecast is :
# rolling_predictions = pd.Series(rolling_predictions, index=returns.index[-1000:])
#
# # And finally the comparison with the realized volatility:
# start='2011-12-06'
# predicted_vol=rolling_predictions[start:].diff().to_frame()
# realized_vol=returns[start:].abs().diff()
# test=pd.merge(predicted_vol,realized_vol,on='Date')[start:]     # merge both DataFrames
# test['Signal1']=np.where((test[0]>0)&(test['SPY']>0),1,0)       # Check the same sign
# test['Signal2']=np.where((test[0]<0)&(test['SPY']<0),1,0)
# result = (test['Signal1'].sum() + test['Signal2'].sum())/test['Signal1'].count()  # count the number of times the sign is the same
# print(result)


if __name__ == "__main__":
    pass