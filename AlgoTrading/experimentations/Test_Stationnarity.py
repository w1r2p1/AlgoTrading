from backtesting.ZigZag import *
# import datetime
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as tsa
from statsmodels.graphics.tsaplots import plot_acf
import seaborn as sns
import pandas_ta as ta
from pmdarima.arima.utils import ndiffs

# exchange = Binance(filename='credentials.txt')
# df = exchange.GetPairKlines(pair='LTCBTC', timeframe='1h', start_date=datetime(2020,6,1))[-200:]

quoteasset = 'BTC'
pair       = 'ADABTC'
timeframe  = '1h'
ClosesOrReturns = ''

# Get the dataframe from the csv file
filename = 'Historical_data/' + quoteasset + '/' + pair + '_' + timeframe  + ClosesOrReturns
df = pd.read_csv(filename, sep='\t')

# df.reset_index()
# df.set_index('time')

# Analyze the correlation betwwen the indicators
correlation = df[['EMA_10', 'EMA_20', 'EMA_50']].corr()
# Plot the correlation heatmap using Seaborn
ax = sns.heatmap(correlation, cmap="Blues", vmin=0, vmax=1, square=True)

# Estimate the number of differences to apply using an ADF test (to get stationnay data) :
n_adf = ndiffs(df['close'], test='adf')  # -> 1

# # Make the time-series stationnary
df['diffed'] = df['close'].diff()
df['logged_and_diffed']  = np.log(df['close']) - np.log(df['close'].shift(1))  # Not usefull in this case
df['pandas_log_returns'] = ta.log_return(df.close)

# Run Augmented Dickey-Fuller test to check if the time-series is stationnary
result = tsa.adfuller(df['logged_and_diffed'].values[1:], autolag='AIC')
print('logged_and_diffed : p-value : %f' % result[1])                                           # if p < 0.005, the time-series is stationnary


# # ______________________ PLOTS _____________________________
# Check if the data is stationnary.
# Original Series
fig, axes = plt.subplots(3, 2)
axes[0, 0].plot(df.close)
axes[0, 0].set_title('Original Series')
plot_acf(df.close, ax=axes[0, 1])

# Diffed
axes[1, 0].plot(df.diffed)
axes[1, 0].set_title('1st Order Differencing')
plot_acf(df.diffed.dropna(), ax=axes[1, 1], lags=80)

# Logged and diffed
axes[2, 0].plot(df.logged_and_diffed)
axes[2, 0].set_title('1st Order Differencing and log')
plot_acf(df.logged_and_diffed.dropna(), ax=axes[2, 1], lags=80)
plt.show()

# plot_acf(df.logged_and_diffed.dropna())
# plt.show()






if __name__ == "__main__":
    pass