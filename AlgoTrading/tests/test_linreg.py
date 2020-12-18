import pandas as pd
import numpy as np
import pandas_ta as ta
from Exchange     import Binance
from decimal  import Decimal, getcontext
import matplotlib.pyplot as plt


class LinReg:

    def __init__(self, timeframe):
        self.exchange  = Binance(filename='credentials.txt')
        self.timeframe = timeframe
        self.df        = pd.DataFrame()


    def prepare_df(self, quote:str, pair:str):

        min_ = 0
        # min_ = 6000
        # max_ = None
        max_ = 200

        # Get the dataframes from the csv files, keep only the time and close columns
        df_hrs_ = pd.read_csv(f'historical_data/{quote}/{self.timeframe}/{pair}_{self.timeframe}', sep='\t').loc[:,['time', 'close']]   # .iloc[min_:max_]
        df_min_ = pd.read_csv(f'historical_data/{quote}/1m/{pair}_1m', sep='\t').loc[:,['time', 'close']]       # .iloc[min_:max_]
        # Rename the close columns to the pair's name
        df_hrs_.columns = ['time', pair+'_h']
        df_min_.columns = ['time', pair+'_m']
        # Set indexes
        df_hrs_.set_index('time', inplace=True)
        df_min_.set_index('time', inplace=True)

        # Test : if data of 2 timeframes match
        # print(df_hrs_.loc['2017-12-28 12:00:00.000'])
        # print(df_min_.loc['2017-12-28 12:00:00'])

        # Remove duplicated lines in the historical data if present
        df_hrs = df_hrs_.loc[~df_hrs_.index.duplicated(keep='first')]
        df_min = df_min_.loc[~df_min_.index.duplicated(keep='first')]

        # Reformat datetime index, Binance's data is messy
        df_hrs.index = pd.to_datetime(df_hrs.index, format='%Y-%m-%d %H:%M:%S.%f')
        df_min.index = pd.to_datetime(df_min.index, format='%Y-%m-%d %H:%M:%S.%f')

        # Merge in a single dataframe that has the time as index
        self.df = pd.merge(df_hrs, df_min, how='outer', left_index=True, right_index=True)

        # If the two coins don't have enough overlapping, skip the pair.
        if len(self.df.dropna()) < 800:
            print(f'{pair} - Less than 1000 ({len(self.df.dropna())}) matching indexes, skipping the pair.')
            return

        # To simplify the code, shift the next minute data 1 place backwards so that the indice of the next minute candle matches the hours' one that gives the signal.
        # self.df.loc[:,pair+'_m'] = self.df.loc[:,pair+'_m'].shift(-1)

        # Drop all the non-necessary minute data : since we shifted, drop averythime at non hours indexes, where hours data is at NaN
        self.df.dropna(inplace=True)

        self.df = self.df.iloc[min_:max_]

        length = 30
        self.df.ta.linreg(close=self.df.loc[:,pair+'_h'], length=length, append=True, tsf=False, col_names=('linreg_'+str(length),))
        self.df.ta.cfo(close=self.df.loc[:,pair+'_h'], length=length, append=True)


        # Plot ______________________________________________________________________________________________________________________
        fig, ax1 = plt.subplots(figsize = (14,8))
        ax2 = ax1.twinx()
        ax1.plot(self.df.index, self.df.loc[:, pair+'_h'], color='blue', label='price')
        ax1.plot(self.df.index, self.df.loc[:, 'linreg_'+str(length)], color='green', label='linreg')
        ax2.plot(self.df.index, self.df.loc[:, 'CFO_'+str(length)],       color='yellow')
        ax1.set_title(f'LTCBTC   -   LinearRegression  -  {self.timeframe}    -    length={length}\n')
        ax1.set_ylabel(f'LTCBTC Price')
        ax2.tick_params(axis='y',  colors='red')
        ax2.axhline(0, color='red', linestyle='--')
        ax1.legend(loc="upper left")
        plt.subplots_adjust(hspace=10)
        plt.show()



if __name__ == '__main__':

    linear_regression = LinReg('2h')

    linear_regression.prepare_df('BTC', 'LTCBTC')