import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot
from plotly.subplots import make_subplots
from stocktrends import indicators
import matplotlib.pyplot as plt
import pandas_ta as ta




if __name__ == '__main__':

    df = pd.read_csv(f'../historical_data/BTC/1h/ETHBTC_1h', sep='\t')
    df.set_index('time', inplace=True)
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S.%f')
    df.drop(columns=['buys', 'sells'], inplace=True)
    df = df.iloc[int(len(df)*0.8):,:]

    df.ta.true_range(high=df.loc[:, 'high'], low=df.loc[:, 'low'], close=df.loc[:, 'close'], append=True)


    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(14,12))
    ax2 = ax1.twinx()
    ax1.plot(df.index, df.loc[:, 'close'], color='blue')
    ax2.plot(df.index, df.loc[:, f'TRUERANGE_{1}'], color='orange')
    ax3.plot(df.index, df.loc[:, f'TRUERANGE_{1}'], color='orange')
    plt.show()
