import pandas as pd
import numpy as np
import pandas_ta as ta
from Exchange     import Binance
from decimal  import Decimal, getcontext
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class Squeeze:

    def __init__(self, timeframe):
        self.exchange  = Binance(filename='credentials.txt')
        self.timeframe = timeframe
        self.df        = pd.DataFrame()


    def prepare_df(self, quote:str, pair:str):

        min_ = 0
        # min_ = 6000
        # max_ = None
        max_ = 500

        # Get the dataframes from the csv files, keep only the time and close columns
        df_hrs_ = pd.read_csv(f'historical_data/{quote}/{self.timeframe}/{pair}_{self.timeframe}', sep='\t').loc[:,['time', 'high', 'low', 'close']]   # .iloc[min_:max_]
        df_min_ = pd.read_csv(f'historical_data/{quote}/1m/{pair}_1m', sep='\t').loc[:,['time', 'high', 'low', 'close']]       # .iloc[min_:max_]
        # Rename the close columns to the pair's name
        df_hrs_.columns = ['time', pair+'_high_h', pair+'_low_h', pair+'_close_h']
        df_min_.columns = ['time', pair+'_high_m', pair+'_low_m', pair+'_close_m']
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

        # Drop all the non-necessary minute data
        self.df.dropna(inplace=True)

        self.df = self.df.iloc[min_:max_]

        length = 30
        self.df.ta.squeeze(high   = self.df.loc[:,pair+'_high_h'],
                           low    = self.df.loc[:,pair+'_low_h'],
                           close  = self.df.loc[:,pair+'_close_h'],
                           bb_std = 3,
                           append = True,
                           detailed = True,
                           lazybear = True,
                           )

        print(self.df.columns)

        squeezeON_on_price  = np.where(self.df.loc[:,'SQZ_ON']==1,  self.df.loc[:,pair+'_close_h'], np.nan)
        squeezeOFF_on_price = np.where(self.df.loc[:,'SQZ_OFF']==1, self.df.loc[:,pair+'_close_h'], np.nan)

        # Pairs and spread ______________________________________________________________________________________________________________________
        max_indice = max_
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14,12))
        # First plot
        ax1.plot(self.df.index, self.df[pair+'_close_h'], color='blue',  label=f"{pair.replace(quote, '')} price in {quote}")
        ax1.plot(self.df.index, squeezeON_on_price,  color='red',  label=f"Squeeze ON")
        ax1.plot(self.df.index, squeezeOFF_on_price, color='black',  label=f"Squeeze OFF")
        # ax1.plot(self.df.index[:max_indice], self.df[pair+'_high_h'].iloc[:max_indice],  color='black')
        # ax1.plot(self.df.index[:max_indice], self.df[pair+'_low_h'].iloc[:max_indice],   color='magenta')
        # Legend and tites
        ax1.set_title(f'Squeeze  -  {self.timeframe}\n\nPrices of {pair.replace(quote, "")} in {quote}')
        ax1.legend(loc="upper left")
        ax1.set_ylabel(f'Price of {pair.replace(quote, "")} in {quote}')
        ax1.tick_params(axis='y',  colors='blue')
        # Second plot
        ax4 = ax2.twinx()
        ax2.plot(self.df.index[:max_indice], self.df['SQZ_20_3.0_20_1.5_LB'].iloc[:max_indice],  color='blue', label="SQZ_20_2.0_20_1.5")

        ax2.plot(self.df.index[:max_indice], self.df['SQZ_INC'].iloc[:max_indice], color='green',  label="SQZ_INC")
        ax2.plot(self.df.index[:max_indice], self.df['SQZ_DEC'].iloc[:max_indice], color='red',    label="SQZ_DEC")
        # ax2.plot(self.df.index[:max_indice], self.df['SQZ_PINC'].iloc[:max_indice], color='black',   label="SQZ_PINC")
        # ax2.plot(self.df.index[:max_indice], self.df['SQZ_PDEC'].iloc[:max_indice], color='yellow',   label="SQZ_PDEC")
        # ax2.scatter(self.df.index[:max_indice], self.df['SQZ_NDEC'].iloc[:max_indice], color='red',   label="SQZ_NDEC")
        # ax2.scatter(self.df.index[:max_indice], self.df['SQZ_NINC'].iloc[:max_indice], color='green',   label="SQZ_NINC")
        ax4.scatter(self.df.index[:max_indice], self.df['SQZ_ON'].iloc[:max_indice].replace({0:np.nan}),  color='green',  label="SQZ_ON")
        ax4.scatter(self.df.index[:max_indice], self.df['SQZ_OFF'].iloc[:max_indice].replace({0:np.nan}), color='black',  label="SQZ_OFF")
        ax2.set_title(f'Squeeze')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Squeeze')
        ax4.set_ylabel('Squeeze ON/OFF')
        ax2.legend(loc="upper left")
        ax2.grid(linestyle='--', axis='y')
        plt.subplots_adjust(hspace = 10)
        plt.show()



if __name__ == '__main__':

    squeeze = Squeeze('2h')

    squeeze.prepare_df('BTC', 'LTCBTC')