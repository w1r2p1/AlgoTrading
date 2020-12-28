# Implementation of algorithm from http://stackoverflow.com/a/22640362/6029703
import numpy as np
import pylab
import pandas as pd
import pandas_ta as ta

# https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data

def thresholding_algo(y, lag, threshold, influence):
    signals   = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag-1] = np.mean(y[0:lag])
    stdFilter[lag-1] = np.std(y[0:lag])

    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter[i-1]:
            if y[i] > avgFilter[i-1]:
                signals[i] = 1
            else:
                signals[i] = -1

            filteredY[i] = influence * y[i] + (1-influence) * filteredY[i-1]
            avgFilter[i] = np.mean(filteredY[(i-lag):i])
            stdFilter[i] = np.std(filteredY[(i-lag):i])
        else:
            signals[i] = 0
            # Adjust the filters
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i-lag):i])
            stdFilter[i] = np.std(filteredY[(i-lag):i])

    return dict(signals   = np.asarray(signals),
                avgFilter = np.asarray(avgFilter),
                stdFilter = np.asarray(stdFilter))


if __name__ == "__main__":

    first_index = -1500


    # Get the historical data from the csv file
    historical_data_file = 'historical_data/{quote}/{pair}_{timeframe}_log'.format(quote='BTC', pair='ADABTC', timeframe='1h')
    df = pd.read_csv(historical_data_file, sep='\t')[first_index:]
    del df['time']

    # y = np.array(df['open_log_returns'])
    y = np.array(ta.ema(df['open_log_returns'], length=5))
    # y = np.array(df['open'])
    # y = np.array(ta.ema(df['open'], length=5))


    # Settings
    lag = 200
    threshold = 5
    influence = 0

    # # Run algo with settings from above
    # result = thresholding_algo(y, lag=lag, threshold=threshold, influence=influence)


    # Calcul des triggers sur les logs
    result = thresholding_algo(y, lag=lag, threshold=threshold, influence=influence)
    # Traduction en prix
    dfraw = pd.read_csv('historical_data/{quote}/{pair}_{timeframe}'.format(quote='BTC', pair='ADABTC', timeframe='1h'), sep='\t')[first_index:]
    dfraw['backtest_buys']  = np.where(result['signals']==-1, dfraw['close'], np.nan)
    dfraw['backtest_sells'] = np.where(result['signals']==1,  dfraw['close'], np.nan)




    # Plot result
    pylab.figure(figsize = (16,12))
    pylab.subplot(311)
    pylab.step(np.arange(1, len(y)+1), dfraw['close'], lw=1.5)
    pylab.step(np.arange(1, len(y)+1), dfraw['backtest_buys'],  color="red",   lw=5)
    pylab.step(np.arange(1, len(y)+1), dfraw['backtest_sells'], color="green", lw=5)

    pylab.subplot(312)
    pylab.plot(np.arange(1, len(y)+1), y)

    pylab.plot(np.arange(1, len(y)+1),
               result["avgFilter"], color="cyan", lw=2)

    pylab.plot(np.arange(1, len(y)+1),
               result["avgFilter"] + threshold * result["stdFilter"], color="green", lw=1.5)

    pylab.plot(np.arange(1, len(y)+1),
               result["avgFilter"] - threshold * result["stdFilter"], color="green", lw=1.5)

    pylab.subplot(313)
    pylab.step(np.arange(1, len(y)+1), result["signals"], color="red", lw=1.5)
    pylab.ylim(-1.5, 1.5)
    pylab.show()
