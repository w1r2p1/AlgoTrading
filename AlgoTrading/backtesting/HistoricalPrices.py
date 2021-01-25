from Exchange   import Binance
from Database   import BotDatabase
from backtesting import ZigZag

from datetime import datetime
from pathlib  import Path
import numpy as np
import pandas as pd


class Histories:
    """ Class to get, update and store the historical prices of all the pairs on Binance. """

    def __init__(self):
        self.database  = BotDatabase("../assets/database_paper.db")
        self.exchange  = Binance(filename='../assets/credentials.txt')
        self.timeframe = '1m'
        self.existing_quoteassets = list(set([dict(bot)['quote'] for bot in self.database.get_all_bots()]))             # ['ETH', 'BTC']


    def GetAllHistories(self):
        """ Loop through all the quotes and paris and store their historical prices until now. """

        pairs = self.exchange.GetNameOfPairs_WithQuoteasset(self.existing_quoteassets)                                    # {'ETH': ['QTUMETH', 'EOSETH',..], 'BTC':[]}

        # for quote in pairs.keys():
        for quote in ['ETH']:
            # In the project directory, create a nested directory for the quoteasset if not exists
            Path(f'historical_data/{quote}/{self.timeframe}').mkdir(parents=True, exist_ok=True)

            # self.GetAndStore_HistoricalPrices(quote = quote, pair='ETHBTC')

            # for counter, pair in enumerate(pairs[quote]):
            for counter, pair in enumerate(['LTCETH', 'ADAETH']):
                # for counter, pair in tqdm(enumerate(["XRPBTC", "ETHBTC"])):
                print(f"\nGETTING DATA FOR {pair}. {counter} out of {len(pairs[quote])}.")
                self.GetAndStore_HistoricalPrices(quote=quote, pair=pair)


    def GetAndStore_HistoricalPrices(self, quote:str, pair:str, **kwargs):
        """ Retrieves the prices history for a pair since is listing on Binance, until now. """

        try:
            kwargs['disable_tqdm'] = False
            df_ = self.exchange.GetPairKlines(pair=pair, timeframe=self.timeframe, start_date=datetime(2018,1,1), **kwargs)

            # Remove duplicated lines in the historical data if present
            df = df_.loc[~df_.time.duplicated(keep='first')]
            # Reformat datetime index, Binance's data is messy
            df.time = pd.to_datetime(df.time, format='%Y-%m-%d %H:%M:%S.%f')

            # # Test comparaison valeurs across timeframes sur les dernières live candles
            # df_test_hrs = self.exchange.GetPairKlines(pair=pair, timeframe='2h', candles=5,   **kwargs).loc[:,['time', 'close']]
            # df_test_min = self.exchange.GetPairKlines(pair=pair, timeframe='1m', candles=650, **kwargs).loc[:,['time', 'close']]
            # # Rename the close columns to the pair's name
            # df_test_hrs.columns = ['time', pair+'_h']
            # df_test_min.columns = ['time', pair+'_m']
            # # Set indexes
            # df_test_hrs.set_index('time', inplace=True)
            # df_test_min.set_index('time', inplace=True)
            # # Reformat datetime index, Binance's data is messy
            # df_test_hrs.index = pd.to_datetime(df_test_hrs.index, format='%Y-%m-%d %H:%M:%S.%f')
            # df_test_min.index = pd.to_datetime(df_test_min.index, format='%Y-%m-%d %H:%M:%S.%f')
            #
            # df_test_tot = pd.concat([df_test_hrs, df_test_min], axis='columns')
            #
            # # print(df_test_tot)
            # # print(df_test_tot.dropna())
            #
            # # Test : if data of 2 timeframes match
            # print(df_test_hrs.loc['2020-09-26 06:00:00'])
            # print(df_test_min.loc['2020-09-26 06:00:00'])
            #
            # return

            df_ret = df.copy()
            df_log = df.copy()
            print(f"Retrieved {len(df)} candles for {pair}.")

            # csv file name, in the corresponding quote folder.
            filename = f'../historical_data/{quote}/{self.timeframe}/{pair}_{self.timeframe}'

            """ _____________________________________________________________________________ """
            """ Work on the close prices """

            # Add the best buy&sell points possible
            """ _______________ Archive : Algo maison précédent  & Stackoverflow ____________ """
            # Identify points of interest : best buy&sell positions possible, to serve as reference points.
            # shift = 1
            # df['buys']  = df.close[(df.close.shift(shift) > df.close) & (df.close.shift(-shift) > df.close)]       # local minima
            # df['sells'] = df.close[(df.close.shift(shift) < df.close) & (df.close.shift(-shift) < df.close)]       # local maxima

            # Delete the first sell if there is no buy before,
            # otherwise the threshold is computed sell->buys and not buys->sells.
            # buys_first_nonNan  = df['buys'].first_valid_index()
            # sells_first_nonNan = df['sells'].first_valid_index()
            # if sells_first_nonNan < buys_first_nonNan:
            #     df.iloc[sells_first_nonNan, df.columns.get_loc('sells')]  = float('nan')

            # Filter the extrema by pairs, remove those who are vertically too close
            # threshold = 1                                                                         # Min percentage gain between the min, max pair
            # points    = df.dropna(subset=['buys', 'sells'], how='all').copy()
            # ddf       = pd.merge(points['buys'].dropna().reset_index(),
            #                      points['sells'].dropna().reset_index(),
            #                      left_index=True,
            #                      right_index=True)
            # ddf = ddf[(ddf['sells']/ddf['buys'] - 1)*100 > threshold]

            # # Try to set a minimum horizontal distance between the triggers
            # ddf = ddf[ddf['index_x'] > (ddf['index_y']+1)]
            # ddf = ddf[ddf['index_x'] < (ddf['index_y']-1)]                    # a sell cannot happen on the candle right after a buy
            # ddf = ddf[ddf['index_x'] > (ddf.index_y.shift(-1)+1)]             # a buy cannot happen on the candle right after a sell

            # Merge this back onto the original dataframe
            # df['buys']  = df.index.map(ddf.set_index('index_x')['buys'])
            # df['sells'] = df.index.map(ddf.set_index('index_y')['sells'])

            # Delete the last buy if there is no sell after
            # buys_last_nonNan  = df['buys'].last_valid_index()
            # sells_last_nonNan = df['sells'].last_valid_index()
            # if buys_last_nonNan > sells_last_nonNan:
            #     df.iloc[buys_last_nonNan, df.columns.get_loc('buys')]  = float('nan')

            # Plot results
            # plt.scatter(ddf['index_x'], ddf['buys'],  c='r')
            # plt.scatter(ddf['index_y'], ddf['sells'], c='g')
            # plt.scatter(df.index, df['buys'],  c='r')
            # plt.scatter(df.index, df['sells'], c='g')
            # df.close.plot()
            # plt.show()

            # """ _______________ Archive : StackOverflow _______________________ """
            # # Generate a noisy AR(1) sample
            # np.random.seed(0)
            # rs = np.random.randn(200)
            # xs = [0]
            # for r in rs:
            #     xs.append(xs[-1]*0.9 + r)
            # df = pd.DataFrame(xs, columns=['data'])
            #
            # df.set_index(pd.Index([i for i in range(0, len(df))]), inplace=True)
            #
            # # Find local peaks
            # df['min'] = df.data[(df.data.shift(1) > df.data) & (df.data.shift(-1) > df.data)]
            # df['max'] = df.data[(df.data.shift(1) < df.data) & (df.data.shift(-1) < df.data)]

            # # # LES 2 METHODES DONNENT LE MEME RESULTAT

            # # # METHODE PERSO -----------------------------------------------------------------

            # Filter the local peaks < threshold, by min, max pairs
            # threshold = 1
            # for j in range(0, len(df['min'])):
            #     if pd.notna(df['min'][j]):
            #         k = j
            #         while pd.isna(df['max'][k]) and k < len(df['min'])-1:
            #             k += 1
            #
            #         # if abs((df['max'][k]/df['min'][j]-1)*100) < 50:
            #         if df['max'][k]-df['min'][j] < threshold:
            #             df.iloc[j, df.columns.get_loc('min')] = float('nan')
            #             df.iloc[k, df.columns.get_loc('max')] = float('nan')
            #
            # # Plot results
            # plt.scatter(df.index, df['min'], c='r')
            # plt.scatter(df.index, df['max'], c='g')
            # df.data.plot()
            # plt.show()


            # # # METHODE STACKOVERFLOW -----------------------------------------------------------------

            # # Delete the first sell if there is no buy before.
            # # Necessary to have both methods yield the same result, otherwise the threshold is computed sell->buys and not buys->sells.
            # min_first_nonNan = df['min'].first_valid_index()
            # max_first_nonNan = df['max'].first_valid_index()
            # if max_first_nonNan < min_first_nonNan:
            #     df.iloc[max_first_nonNan, df.columns.get_loc('max')]  = float('nan')
            #
            # threshold = 1
            # points = df.dropna(subset=['min', 'max'], how='all').copy()     # Remove missing values in 'min' & 'max' columns
            # ddf    = pd.merge(points['min'].dropna().reset_index(),
            #                   points['max'].dropna().reset_index(),
            #                   left_index=True,
            #                   right_index=True)
            # ddf    = ddf[ddf['max'] > (ddf['min'] + threshold)]
            #
            # # Plot results
            # plt.scatter(ddf['index_x'], ddf['min'], c='r')
            # plt.scatter(ddf['index_y'], ddf['max'], c='g')
            # df.data.plot()
            # plt.show()

            """"""
            ZigZag.Add_ZigZag(df, min_perc_change=4)
            # df.close.plot()
            # plt.scatter(df.index, df['buys'],  c='r')
            # plt.scatter(df.index, df['sells'],  c='g')
            # plt.show()

            df.to_csv(filename, sep='\t', index=False, na_rep="NaN")                                      # index=False drops the index (the column is not written in the file).


            """ _____________________________________________________________________________ """
            """ Repeat with the  returns (the percentage the price has moved) """

            # https://mlfinlab.readthedocs.io/en/latest/labeling/labeling_raw_return.html
            # returns = df.close/df.close.shift(-1) - 1
            #         = df.close.pct_change()
            # log ret = np.log(df.close/df.close.shift(-1))
            #         = np.log(df.close.pct_change() + 1)
            #         = np.log(returns + 1)

            df_ret['open_returns']  = df.open.pct_change()
            df_ret['high_returns']  = df.high.pct_change()
            df_ret['low_returns']   = df.low.pct_change()
            df_ret['close_returns'] = df.close.pct_change()

            # # Add the best buy&sell points possible, based on those detected on the close prices.
            # # Add_ZigZag_pct_change(df_pct, min_change=4)
            df_ret['buys']  = np.where(df['buys']==df['close'],  df_ret['close_returns'], float('nan'))      # Conditionally fill column values based on another columns value.
            df_ret['sells'] = np.where(df['sells']==df['close'], df_ret['close_returns'], float('nan'))

            # df_ret.to_csv(filename + '_ret', sep='\t', index=False, na_rep="NaN")




            """ _____________________________________________________________________________ """
            """ Repeat with the log returns (the percentage the price has moved in log scale) """

            # https://mlfinlab.readthedocs.io/en/latest/labeling/labeling_raw_return.html
            # returns = df.close/df.close.shift(-1) - 1
            #         = df.close.pct_change()
            # log ret = np.log(df.close/df.close.shift(-1))
            #         = np.log(df.close.pct_change() + 1)
            #         = np.log(returns + 1)

            df_log['open_log_returns']  = np.log(df_log.open.pct_change()  + 1)
            df_log['high_log_returns']  = np.log(df_log.high.pct_change()  + 1)
            df_log['low_log_returns']   = np.log(df_log.low.pct_change()   + 1)
            df_log['close_log_returns'] = np.log(df_log.close.pct_change() + 1)

            # We work with log returns, we don't need the raw prices
            del df_log['open']; del df_log['high']; del df_log['low']; del df_log['close']

            # Add the best buy&sell points possible, based on those detected on the close prices.
            # Add_ZigZag_pct_change(df_pct, min_change=4)
            df_log['buys']  = np.where(df['buys']==df['close'],  df_log['close_log_returns'], float('nan'))      # Conditionally fill column values based on another columns value.
            df_log['sells'] = np.where(df['sells']==df['close'], df_log['close_log_returns'], float('nan'))

            # df_log.to_csv(filename + '_log', sep='\t', index=False, na_rep="NaN")


        except Exception as e:
            print("Could not process data on " + pair + ". Skipping the pair. Error : ")
            print(e)


    def UpdatePairHistory(self, quote:str, pair:str):
        """ Updates the CSV file containing the historical data with new recent data until now. """

        # csv file name, in the corresponding quote folder.
        filename = 'Historical_data/' + quote + '/' + pair + '_'+ self.timeframe

        # Get what is in the file right now
        stored_data = pd.read_csv(filename, sep='\t')
        # Get the last stored date in datetime format
        last_stored_date = datetime.strptime(stored_data.iloc[-1, stored_data.columns.get_loc('time')], '%Y-%m-%d %H:%M:%S')
        # Get the candles sinec the last date
        new_data    = self.exchange.GetPairKlines(pair=pair, timeframe=self.timeframe, start_date=last_stored_date)
        # Append the new data to the old data
        data_to_store = stored_data.append(new_data)

        with open(filename, 'w'):
            # Re-write the whole data to the CSV file
            data_to_store.to_csv(filename, sep='\t', index=False, na_rep="NaN")                                       # index=False drops the index (the column is not written in the file).
            # Append the new data to the CSV file
            # new_data.to_csv(filename, mode='a', sep='\t', header=False, index=False, na_rep="NaN")




if __name__ == "__main__":
    history = Histories()
    history.GetAllHistories()
    # history.GetAndStore_HistoricalPrices('BTC', 'ADABTC')
    # history.UpdatePairHistory('BTC', 'ADABTC')