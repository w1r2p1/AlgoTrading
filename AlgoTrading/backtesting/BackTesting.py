# from Exchange import Binance
from Database import BotDatabase
from Helpers import HelperMethods

from decimal  import Decimal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from empyrical import sortino_ratio, calmar_ratio, omega_ratio, sharpe_ratio
import pandas_ta as ta
import datetime
import itertools
from   multiprocessing.pool import ThreadPool as Pool
import time
import plotly.graph_objs as go
from plotly.offline import plot
from plotly.subplots import make_subplots
import optuna
import sys
import re


class BackTesting:

    def __init__(self,
                 timeframe,
                 quote:str,
                 pair:str,
                 strategy_name:str,
                 starting_balances:dict,
                 alloc_pct:int,
                 stop_loss_pct,
                 plot:bool=False,
                 **kwargs):
        self.database  = BotDatabase(name="../assets/database_paper.db")
        self.helpers   = HelperMethods(database=self.database)
        self.df        = pd.DataFrame()

        self.timeframe 			= timeframe
        self.quote 				= quote
        self.pair 				= pair
        self.base 				= pair.replace(quote, '')
        self.strategy_name		= strategy_name
        self.starting_balances 	= starting_balances
        self.alloc_pct 			= alloc_pct
        self.stop_loss_pct 		= stop_loss_pct
        self.plot 				= plot
        self.quote_profits      = 0
        self.base_profits       = 0
        self.buy_hold           = 0
        self.trailing_stop      = Decimal(0)
        self.display_progress_bar = kwargs.get('display_progress_bar', True)
        self.print_output       = kwargs.get('print_output', True)
        # self.optimize_at_each_candle = kwargs.get('optimize_at_each_candle', {})

        # Create a dictionnary that
        self.indicators_dict = {}
        indic_number = 1
        indic_dict = not None
        while indic_dict is not None:
            indic_dict = kwargs.get(f'indic_{indic_number}', {})	# indic is a dict
            if indic_dict == {}:
                break
            self.indicators_dict[str(indic_number)] = dict(indic           = indic_dict['indicator'],
                                                           indic_name      = f"{indic_dict['indicator']}_{indic_dict['length']}",
                                                           indic_length    = indic_dict['length'],
                                                           indic_close_col = indic_dict['close'])
            indic_number += 1


        if kwargs.get(f'indic_{indic_number}', {})!={}:     # If we have at least one indicator specified
            # Will be used to set the first trading point
            self.longest_length = max([self.indicators_dict[str(i)]['indic_length'] for i in self.indicators_dict.keys()])
        else :
            self.longest_length = 1

        self.rsi_lower_threshold = kwargs.get('rsi_lower_threshold', None)
        self.rsi_upper_threshold = kwargs.get('rsi_upper_threshold', None)

    def prepare_df(self):

        # Get the dataframes from the csv files, keep only specific columns
        df_hrs_ = pd.read_csv(f'../historical_data/{self.quote}/{self.timeframe}/{self.pair}_{self.timeframe}', sep='\t').loc[:,['time', 'open', 'high', 'low', 'close', 'volume']]
        df_hrs_.columns = ['time', 'open_h', 'high_h', 'low_h', 'close_h', 'volume_h']
        df_min_ = pd.read_csv(f'../historical_data/{self.quote}/1m/{self.pair}_1m', sep='\t').loc[:,['time', 'close']]
        # Rename the close columns to the pair's name
        df_min_.columns = ['time', 'close_m']
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
            print(f'{self.pair} - Less than 1000 ({len(self.df.dropna())}) matching indexes, skipping the pair.')
            return

        # To simplify the code, shift the next minute data 1 place backwards so that the indice of the next minute candle matches the hours' one that gives the signal.
        # self.df.loc[:,'close_m'] = self.df.loc[:,'close_m'].shift(-1)

        # Drop all the non-necessary minute data : since we shifted, drop averythime at non hours indexes, where hours data is at NaN
        # print('Data is ready.')
        self.df.dropna(inplace=True)


    def compute_indicators(self):

        if self.strategy_name == 'Mean_Reversion_spread':
            self.df.loc[:,'spread'] = np.log(self.df.loc[:,'close_h'].pct_change() + 1)

        for indic_dict in self.indicators_dict.values():
            indic           = indic_dict['indic']
            indic_name      = (indic_dict['indic_name'],) if indic!='bbands' else ('bband_l', 'bband_m', 'bband_u')
            indic_length    = indic_dict['indic_length']
            indic_close_col = indic_dict['indic_close_col']

            close_ = self.df.loc[:,'close_h'] if indic_close_col=='close' else indic_close_col
            open_  = self.df.loc[:,'open_h']  if 'open_h' in self.df.columns else None
            high_  = self.df.loc[:,'high_h']  if 'high_h' in self.df.columns else None
            low_   = self.df.loc[:,'low_h']   if 'low_h'  in self.df.columns else None
            getattr(self.df.ta, indic)(open=open_, high=high_, low=low_, close=close_, length=indic_length, append=True, col_names=indic_name)


    def update_best_parameters(self, start_index:int, end_index:int, n_trials:int):
        """ Runs an Optuna optimzation to find the best parameters over a specified period. """

        best_parameters_dict, best_value = OptunaOptimization(timeframe     = self.timeframe,
                                                              quote         = self.quote,
                                                              pair          = self.pair,
                                                              strategy_name = self.strategy_name,
                                                              readjust_best_parameters = False,
                                                              start_index   = start_index,
                                                              end_index     = end_index,
                                                              ).run_optuna_optimization(n_trials=n_trials, show_progress_bar=False)             # best_parameters = {'length_1': 8, 'stop_loss_pct': 3, 'rsi_lower_threshold': 49}

        # Update the parameters only if the best trial is > 0
        if best_value > 0:
            best_parameters = list(best_parameters_dict.items())                                        # Transform the dict in a list of tuples : [(key, value), ...]

            # Modify the values of the dict storing the indicators
            for key, indic_dict in self.indicators_dict.items():
                old_indic_name = indic_dict['indic_name']
                new_indic_length = best_parameters[int(key)][1]
                self.indicators_dict[key]['indic_name'] = f"{indic_dict['indic']}_{new_indic_length}"
                self.indicators_dict[key]['indic_length'] = new_indic_length
                self.df.drop(columns=[old_indic_name])

        self.compute_indicators()


    def find_signal(self, i:int)->str:
        """ Compute the indicators and look for a signal.
            Each strategy is only a template that needs to be fine tuned.
        """
        signal = ''

        if self.strategy_name=='Crossover':
            fast         = self.df[self.indicators_dict['1']['indic_name']].iloc[i]
            slow         = self.df[self.indicators_dict['2']['indic_name']].iloc[i]
            fast_shifted = self.df[self.indicators_dict['1']['indic_name']].iloc[i-1]
            slow_shifted = self.df[self.indicators_dict['2']['indic_name']].iloc[i-1]
            signal = 'buy' if fast_shifted < slow_shifted and fast > slow else 'sell' if fast_shifted > slow_shifted and fast < slow else ''

        if self.strategy_name=='MA_slope':
            ma            = self.df[self.indicators_dict['1']['indic_name']].iloc[i]
            slope         = self.df[self.indicators_dict['2']['indic_name']].iloc[i]
            ma_shifted1   = self.df[self.indicators_dict['1']['indic_name']].iloc[i-1]
            slope_shifted = self.df[self.indicators_dict['2']['indic_name']].iloc[i-1]
            signal = 'buy' if slope==0 and ma_shifted1>ma else 'sell' if slope==0 and ma_shifted1<ma else ''

        if self.strategy_name=='Mean_Reversion_simple':
            indic         = self.df[self.indicators_dict['1']['indic_name']].iloc[i]
            price_now     = Decimal(self.df['close_h'].iloc[i])
            price_shifted = Decimal(self.df['close_h'].iloc[i-1])
            signal = 'buy' if price_now<price_shifted and price_now<indic else 'sell' if price_now>price_shifted and price_now>indic  else ''

        if self.strategy_name=='Mean_Reversion_spread':
            spread         = self.df['spread'].iloc[i]
            spread_shifted = self.df['spread'].iloc[i-1]
            bband_l		   = self.df['bband_l'].iloc[i]
            bband_u		   = self.df['bband_u'].iloc[i]
            signal = 'sell' if spread_shifted < spread and spread > bband_u else 'buy' if spread_shifted > spread and spread < bband_l else ''

        if self.strategy_name=='Gap_and_go':
            """ Gap-and-Go strategy :
                buy  when price > high of last red   candle
                sell when price < low  of last green candle
                The indicator can be a filter for the buys and sells. """
            price = self.df['close_h'].iloc[i]
            high_of_last_red_candle  = self.find_high_of_last_red_candle(start_at=i-1)
            low_of_last_green_candle = self.find_low_of_last_green_candle(start_at=i-1)
            signal = 'sell' if price > high_of_last_red_candle else 'buy' if price < low_of_last_green_candle  else ''

        if self.strategy_name=='RSI':
            # self.rsi_upper_threshold = self.rsi_lower_threshold
            rsi         = self.df[self.indicators_dict['1']['indic_name']].iloc[i]
            rsi_shifted = self.df[self.indicators_dict['1']['indic_name']].iloc[i-1]
            # bband_l		   = self.df['bband_l'].iloc[i]
            # bband_u		   = self.df['bband_u'].iloc[i]
            signal = 'buy' if rsi > self.rsi_lower_threshold > rsi_shifted else 'sell' if rsi < self.rsi_upper_threshold < rsi_shifted else ''
            # signal = 'buy' if rsi > bband_l > rsi_shifted else 'sell' if rsi < bband_u < rsi_shifted else ''


        self.df.loc[self.df.index[i],'signal'] = signal
        return signal


    def backtest(self, **kwargs):
        """ Iterates over a specified period to simulate a strategy and computes the outcome.
            Looks for buy/sell signals and places market orders at the triggers.
            To simulate the spread, the order price is the price at the close of the current candle, and the executed price is the close of the candle one minute after the trigger.
            stop loss and trailing stop loss are implemented.
            """

        # For code readability
        pair 				= self.pair
        starting_balances 	= self.starting_balances
        alloc_pct 			= self.alloc_pct
        bot                 = {}
        try:
            bot = dict(self.database.get_bot(pair=pair))
        except Exception as e:
            sys.exit(f"Please create the {pair} bot in the database before backtesting.")

        parsed_timeframe = re.findall(r'[A-Za-z]+|\d+', self.timeframe)      # Separates '30m' in ['30', 'm']
        period = 'last_day' if parsed_timeframe[1]=='m' else 'last_week'
        # candles_for_period = int(60*24/int(parsed_timeframe[0])) if period=='last_day' else int(7*24/int(parsed_timeframe[0])) if period=='last_week' else 1000
        candles_for_period = 200

        # Get the df from the file and prepare it
        self.prepare_df()

        # Limit its size if need be
        min_pct = 0.4 if parsed_timeframe[1]=='m' else 0
        max_pct = 0.8 if parsed_timeframe[1]=='m' else 1
        start_index = kwargs.get('start_index', int(len(self.df.index)*min_pct))
        end_index   = kwargs.get('end_index',   int(len(self.df.index)*max_pct))
        self.df = self.df.iloc[start_index:end_index]

        readjust_best_parameters = kwargs.get('readjust_best_parameters', False)

        # Compute the indicators
        self.compute_indicators()

        # Set the first point
        quote_balance = Decimal(starting_balances['quote'])
        base_balance  = Decimal(starting_balances['base'])
        self.df.loc[self.df.index[0], 'quote_balance'] = quote_balance
        self.df.loc[self.df.index[0], 'base_balance']  = base_balance
        status = 'just sold'		# look to buy first

        # Initialize variables
        trades = dict(nb_trades     = 0,
                      nb_win_trades = 0,
                      nb_los_trades = 0,)

        # Go through all candlesticks
        start = self.longest_length+candles_for_period if readjust_best_parameters else self.longest_length
        end   = len(self.df.loc[:,'close_h'])-1
        data  = range(start, end)
        data  = tqdm(data) if self.display_progress_bar else data      # tqdm : progress bar
        # ____________________________________________________________________________________________________________________________________________________________
        for i in data:

            # Look for a signal
            signal = self.find_signal(i=i)

            price_now = Decimal(self.df['close_h'].iloc[i])
            price_shifted = Decimal(self.df['close_h'].iloc[i-1])

            if status=='just bought': 	# ____________________________________________________________________________________________________________________________
                # Sell either by signal or stop-loss
                price_at_buy = Decimal(self.df.loc[:, 'buyprice_'+pair].dropna().iloc[-1])
                stop_loss_trigger = price_now < self.trailing_stop if kwargs.get('trailing_stop', False) else (price_now/price_at_buy-1)*100 < Decimal(-self.stop_loss_pct)
                increased_more_than_fees = (price_now/price_at_buy-1)*100 > 1

                # Sell the base balance
                if (signal=='sell' and increased_more_than_fees) or stop_loss_trigger:

                    # To simulate the spread, we sell on the following minute candle.
                    price_base_next_minute = Decimal(self.df['close_m'].iloc[i+1])
                    self.df.loc[self.df.index[i], 'sellprice_'+pair] = price_base_next_minute if parsed_timeframe[1]=='h' else price_now
                    # self.df.loc[self.df.index[i], 'sell_on_rsi'] = self.df[self.indicators_dict['1']['indic_name']].iloc[i]

                    base_quantity_to_sell   = self.helpers.RoundToValidQuantity(bot=bot, quantity=base_balance)
                    quote_quantity_sell     = base_quantity_to_sell*price_base_next_minute
                    fee_in_quote_sell       = quote_quantity_sell*Decimal(0.075)/Decimal(100)
                    received_quote_quantity = quote_quantity_sell - fee_in_quote_sell						# What we get in quote from the sell

                    # Update the balances
                    base_balance  -= base_quantity_to_sell
                    quote_balance += received_quote_quantity

                    self.df.loc[self.df.index[i], 'quote_balance'] = quote_balance

                    # Count the (un)sucessfull trades
                    trades['nb_trades'] += 1
                    balance_quote_previous = self.df.loc[:, 'quote_balance'].dropna().iloc[-1]
                    if quote_balance < balance_quote_previous:
                        trades['nb_los_trades'] += 1
                    else:
                        trades['nb_win_trades'] += 1

                    self.df.loc[self.df.index[i], 'fees'] = fee_in_quote_sell
                    status = 'just sold'
                    # if general_loop:
                    #     print('sold')

                    # Look for the best parameters combination over the last week/day
                    if readjust_best_parameters:
                        self.update_best_parameters(start_index=i-(self.longest_length+candles_for_period), end_index=i, n_trials=20)

                # Adjust the trailing stop loss (needs to be activated) if the price has gone up
                else:
                    trailing_stop = price_now*(1-Decimal(self.stop_loss_pct)/100)
                    if price_now > price_shifted and trailing_stop > self.trailing_stop:
                        self.trailing_stop = trailing_stop

            elif status=='just sold': 	# ___________________________________________________________________________________________________________________________
                if signal=='buy':
                    # To simulate the spread, we buy on the following minute candle (it has been shifted already, so it's on the same index).
                    price_base_next_minute = Decimal(self.df['close_m'].iloc[i+1])
                    self.df.loc[self.df.index[i], 'buyprice_'+pair] = price_base_next_minute if parsed_timeframe[1]=='h' else price_now
                    # self.df.loc[self.df.index[i], 'buy_on_rsi'] = self.df[self.indicators_dict['1']['indic_name']].iloc[i]

                    base_quantity_to_buy   = self.helpers.RoundToValidQuantity(bot=bot, quantity=quote_balance/price_now*alloc_pct/100)
                    quote_quantity_buy     = base_quantity_to_buy*price_base_next_minute
                    fee_in_base_buy        = base_quantity_to_buy*Decimal(0.075)/Decimal(100)
                    received_base_quantity = base_quantity_to_buy - fee_in_base_buy						# What we get in base from the buy

                    # Update the balances
                    base_balance  += received_base_quantity
                    quote_balance -= quote_quantity_buy

                    base_balance_previous = self.df.loc[:, 'base_balance'].dropna().iloc[-1]

                    # Count the (un)sucessfull trades
                    trades['nb_trades'] += 1
                    self.df.loc[self.df.index[i], 'base_balance'] = base_balance
                    if base_balance < base_balance_previous:
                        trades["nb_los_trades"] += 1
                    else:
                        trades['nb_win_trades'] += 1

                    self.df.loc[self.df.index[i], 'fees'] = fee_in_base_buy*price_base_next_minute
                    status = 'just bought'
                    # if general_loop:
                    #     print('bought')
                    # Set the stop loss to : buy price - stop_loss_pct
                    self.trailing_stop = price_base_next_minute*(1-Decimal(self.stop_loss_pct)/100)


        if trades['nb_trades'] > 2:
            # Compute profits and compare to buy and hold
            base_balance_initiale  = self.df.loc[:, 'base_balance'].dropna().iloc[0]
            base_balance_finale    = self.df.loc[:, 'base_balance'].dropna().iloc[-1]
            quote_balance_initiale = self.df.loc[:, 'quote_balance'].dropna().iloc[0]
            quote_balance_finale   = self.df.loc[:, 'quote_balance'].dropna().iloc[-1]
            price_base_at_first_basebalance = self.df.loc[self.df['base_balance'] == base_balance_initiale, 'close_h'].iloc[0]
            price_base_at_last_basebalance  = self.df.loc[self.df['base_balance'] == base_balance_finale,   'close_h'].iloc[-1]
            self.quote_profits = round((quote_balance_finale / quote_balance_initiale - 1)*100, 1)
            # self.base_profits  = (base_balance_finale / base_balance_initiale - 1)*100
            self.buy_hold      = (price_base_at_last_basebalance / price_base_at_first_basebalance - 1)*100

            # Max drawdown
            try:
                quotebalance = self.df.loc[:, 'quote_balance'].dropna().astype(float)
                i = np.argmax(np.maximum.accumulate(quotebalance) - quotebalance)   # end of the period
                j = np.argmax(quotebalance[:i])                                     # start of period
                max_drawdown = round((quotebalance[i]/quotebalance[j]-1)*100, 1)
            except Exception as e:
                max_drawdown = -100

            # Remove 0s & NaNs and compute metric
            temp    = self.df.loc[:,'quote_balance'].astype(float)
            cleaned = temp[np.where(temp, True, False)].dropna().pct_change()
            metric  = sharpe_ratio(cleaned, annualization=1)
            # metric  = sortino_ratio(cleaned, annualization=1)
            metric  = metric if abs(metric) != np.inf and not np.isnan(metric) else Decimal(0)

            # Print outputs
            if self.print_output:
                print(f'{self.quote} profits = {self.quote_profits}%')
                # print(f'{self.base} profits = {round(self.base_profits,  1)}%  (buy & hold = {round(self.buy_hold, 1)}%)')
                print(f'Winning trades : {trades["nb_win_trades"]} ({int(trades["nb_win_trades"]/(trades["nb_win_trades"]+trades["nb_los_trades"])*100)}%)')
                print(f'Losing trades  : {trades["nb_los_trades"]} ({int((1-trades["nb_win_trades"]/(trades["nb_win_trades"]+trades["nb_los_trades"]))*100)}%)')
                print(f'Max drawdown : {round(max_drawdown,2)}%')
                print(f'Sharp ratio : {round(metric,2)}.')

            if self.plot:
                self.plot_backtest(trades=trades, metri=metric)
                # self.plotbacktest_plotly(trades=trades, metri=metric)

            return float(self.quote_profits), metric, max_drawdown
        else:
            # print("Strategy didn't buy or sell.")
            return 0, 0, 0


    def plot_backtest(self, **kwargs):

        quote	      = self.quote
        pair	   	  = self.pair
        alloc_pct     = self.alloc_pct
        stoploss      = self.stop_loss_pct
        trades        = kwargs.get('trades', {})
        metric        = kwargs.get('metric', 0)

        min_indice = -1000
        max_indice = None

        # if self.strategy_name == 'Gap_and_go':
        #     fig = go.Figure(data=[go.Candlestick(x 	   = self.df.index[min_indice:max_indice],
        #                                          open  = self.df['open_h'].iloc[min_indice:max_indice],
        #                                          high  = self.df['high_h'].iloc[min_indice:max_indice],
        #                                          low   = self.df['low_h'].iloc[min_indice:max_indice],
        #                                          close = self.df['close_h'].iloc[min_indice:max_indice],),
        #                           go.Scatter(x=self.df.index[min_indice:max_indice],
        #                                      y=self.df['buyprice_'+pair].iloc[min_indice:max_indice],
        #                                      mode='markers',
        #                                      marker=dict(color='orange'),
        #                                      name='buys',),
        #                           go.Scatter(x=self.df.index[min_indice:max_indice],
        #                                      y=self.df['sellprice_'+pair].iloc[min_indice:max_indice],
        #                                      mode='markers',
        #                                      marker=dict(color='black'),
        #                                      name='sells',)
        #                           ])
        #
        #     # Set line and fill colors of candles
        #     # IGNORE THE WARNING
        #     cs = fig.data[0]
        #     cs.increasing.fillcolor  = '#3D9970'    # '#008000'
        #     cs.increasing.line.color = '#3D9970'    # '#008000'
        #     cs.decreasing.fillcolor  = '#FF4136'    # '#800000'
        #     cs.decreasing.line.color = '#FF4136'    # '#800000'
        #
        #     fig['layout'] = {
        #         "margin"    : {"t": 30, "b": 20},
        #         "title"     : f"{pair} - Gap and Go Strategy - {self.timeframe} - {quote} profits: {round(self.quote_profits,1)}% (buy & hold: {round(self.buy_hold,1)}%). Stoploss: {stoploss}%.",
        #         "titlefont" : dict(size=18,color='#a3a7b0'),
        #         # "height"    : 350,
        #         "xaxis": {
        #             "fixedrange"    : False,
        #             "showline"      : True,
        #             "zeroline"      : False,
        #             "showgrid"      : False,
        #             "showticklabels": True,
        #             "rangeslider"   : {"visible": False},
        #             "color"         : "#a3a7b0"},
        #         "yaxis": {
        #             "fixedrange"    : False,
        #             "showline"      : False,
        #             "zeroline"      : False,
        #             "showgrid"      : True,
        #             "showticklabels": True,
        #             "ticks"         : "",
        #             "color": "#a3a7b0"},
        #         "plot_bgcolor" : "#23272c",
        #         "paper_bgcolor": "#23272c"}
        #
        #     plot(fig)
        #
        # elif self.strategy_name == 'RSI':
        #     fig, (ax2, ax3) = plt.subplots(2, 1, figsize=(14,12))
        #
        #     # First plot - Price
        #     ax2.plot(self.df.index[min_indice:max_indice], self.df.loc[:, 'close_h'][min_indice:max_indice], c='blue',  label=f"{pair} price", linestyle='-')
        #     ax2.set_title(f'{self.strategy_name} Strategy  -  {self.timeframe}    -    Stop-loss={stoploss}%    -    Indicators : {[indic_dict["indic_name"] for indic_dict in self.indicators_dict.values()]}    -    alloc={alloc_pct}%    -    Trades : {trades["nb_trades"]}    -    Thresholds={self.rsi_lower_threshold, self.rsi_upper_threshold}\n\nQuantity of coins held')
        #     # Add the buys and sells
        #     ax2.scatter(self.df.index[min_indice:max_indice], self.df['buyprice_'+pair].iloc[min_indice:max_indice],  color='red',    marker='x', s=65, label='Buys')
        #     ax2.scatter(self.df.index[min_indice:max_indice], self.df['sellprice_'+pair].iloc[min_indice:max_indice], color='green',  marker='x', s=65, label='Sells')
        #     ax2.legend(loc="upper left")
        #     ax2.set_ylabel(f'{quote} balance')
        #     ax2.tick_params(axis='y',  colors='blue')
        #
        #     # Second plot - RSI
        #     ax3.plot(self.df.index[min_indice:max_indice], self.df[self.indicators_dict['1']['indic_name']].iloc[min_indice:max_indice], c='black',  label=f"RSI", linestyle='-')
        #     # ax3.plot(self.df.index[min_indice:max_indice], self.df['bband_l'].iloc[min_indice:max_indice], c='blue',  label=f"bband_l", linestyle='-')
        #     # ax3.plot(self.df.index[min_indice:max_indice], self.df['bband_u'].iloc[min_indice:max_indice], c='blue',  label=f"bband_u", linestyle='-')
        #     # Add the buys and sells
        #     # ax3.scatter(self.df.index[min_indice:max_indice], self.df['buy_on_rsi'].iloc[min_indice:max_indice],  color='red',    marker='x', s=65, label='Buys')
        #     # ax3.scatter(self.df.index[min_indice:max_indice], self.df['sell_on_rsi'].iloc[min_indice:max_indice], color='green',  marker='x', s=65, label='Sells')
        #     ax3.set_title(f'RSI')
        #     ax3.legend(loc="upper left")
        #     ax3.set_ylabel(f'RSI value')
        #     ax3.tick_params(axis='y',  colors='black')
        #     ax3.axhline(50, color='blue',  linestyle='--')
        #     # ax3.axhline(self.rsi_lower_threshold, color='black',  linestyle='--')
        #     # ax3.axhline(self.rsi_upper_threshold, color='black',  linestyle='--')
        #     plt.subplots_adjust(hspace=10)
        #     plt.show()
        #
        # else:
        #     fig, ax1 = plt.subplots(figsize=(14,10))
        #     # First plot
        #     ax1.plot(self.df.index[min_indice:max_indice], self.df['close_h'].iloc[min_indice:max_indice], color='black',  label=f"{pair.replace(quote, '')} price in {quote}")
        #     # Add the buys and sells
        #     ax1.scatter(self.df.index[min_indice:max_indice], self.df['buyprice_'+pair].iloc[min_indice:max_indice],  color='red',    marker='x', s=65, label='Buys')
        #     ax1.scatter(self.df.index[min_indice:max_indice], self.df['sellprice_'+pair].iloc[min_indice:max_indice], color='green',  marker='x', s=65, label='Sells')
        #     # Add the indics
        #     for indic_dict in self.indicators_dict.values():
        #         indic_name      = indic_dict['indic_name']
        #         ax1.plot(self.df.index[min_indice:max_indice], self.df[indic_name].iloc[min_indice:max_indice],   label=indic_name)
        #     # Legend and tites
        #     ax1.set_title(f'{self.strategy_name} Strategy  -  {self.timeframe}\n\nPrices of {pair.replace(quote, "")} in {quote}')
        #     ax1.legend(loc="upper left")
        #     ax1.set_ylabel(f'Price of {pair.replace(quote, "")} in {quote}')
        #     ax1.tick_params(axis='y',  colors='blue')
        #     ax1.grid(linestyle='--', axis='y')
        #     plt.show()
        #

        """Plotly plot """
        # self.plot_price_and_indicators()

        # Strategy evolution ______________________________________________________________________________________________________________________
        fig, (ax2, ax3) = plt.subplots(2, 1, figsize=(14,12))
        ax4 = ax2.twinx()
        ax5 = ax3.twinx()

        # First plot - Balance changes
        quote_mask = np.isfinite(self.df.loc[:, 'quote_balance'].replace({0:np.nan}).astype(np.double))		# mask the NaNs for the plot, to allow to use plot with nan values
        base_mask  = np.isfinite(self.df.loc[:, 'base_balance'].replace({0:np.nan}).astype(np.double))
        ax2.plot(self.df.index[quote_mask], self.df.loc[:, 'quote_balance'][quote_mask], c='blue',  label=f"{quote} balance", linestyle='-', marker='o')
        ax2.plot([], [], 	 					         						         c='green', label=f"{pair.replace(quote, '')} balance", linestyle='-', marker='o')	# Empty plot just for the label
        ax4.plot(self.df.index[base_mask], self.df.loc[:, 'base_balance'][base_mask],    c='green', linestyle='-', marker='o')
        if 'RSI' in self.strategy_name:
            ax2.set_title(f'{self.strategy_name} Strategy  -  {self.timeframe}    -    Stop-loss={stoploss}%    -    Indicators : {[indic_dict["indic_name"] for indic_dict in self.indicators_dict.values()]}    -    alloc={alloc_pct}%    -    Trades : {trades["nb_trades"]}    -    Thresholds={self.rsi_lower_threshold, self.rsi_upper_threshold}\n\nQuantity of coins held')
        else:
            ax2.set_title(f'{self.strategy_name} Strategy  -  {self.timeframe}    -    Stop-loss={stoploss}%    -    Indicators : {[indic_dict["indic_name"] for indic_dict in self.indicators_dict.values()]}    -    alloc={alloc_pct}%    -    Trades : {trades["nb_trades"]}    -    Metric={round(metric,2)}\n\nQuantity of coins held')
        ax2.legend(loc="upper left")
        ax2.set_ylabel(f'{quote} balance')
        ax4.set_ylabel(f'{pair.replace(quote, "")} balance')
        ax2.tick_params(axis='y',  colors='blue')
        ax4.tick_params(axis='y',  colors='green')
        ax2.axhline(self.df['quote_balance'].dropna().iloc[0], color='blue',  linestyle='--')
        ax4.axhline(self.df['base_balance'].dropna().iloc[0], color='green', linestyle='--')

        # Second plot - Percentage changes
        # Deal with NaNs first to allow the use of plot instead of scatter
        self.df.loc[:, 'dquote_balance'] = self.df['quote_balance'].astype(float).pct_change()*100
        self.df.loc[pd.notnull(self.df['quote_balance']) & pd.isnull(self.df['dquote_balance']), 'dquote_balance'] = self.df['quote_balance']
        self.df.loc[:, 'dbase_balance'] = self.df['base_balance'].astype(float).pct_change()*100
        self.df.loc[pd.notnull(self.df['base_balance']) & pd.isnull(self.df['dbase_balance']), 'dbase_balance'] = self.df['base_balance']
        quote_mask = np.isfinite(self.df.loc[:, 'dquote_balance'].replace({0:np.nan}).astype(np.double))		# mask the NaNs for the plot, to allow to use plot with nan values
        base_mask  = np.isfinite(self.df.loc[:, 'dbase_balance'].replace({0:np.nan}).astype(np.double))
        ax3.plot(self.df.index[quote_mask], self.df.loc[:,'dquote_balance'][quote_mask], c='blue',  label=f"{quote} balance % change", linestyle='-', marker='o')
        ax3.plot([], [], 	 					         						         c='green', label=f"{pair.replace(quote, '')} balance % change", linestyle='-', marker='o')	# Empty plot just for the label
        ax5.plot(self.df.index[base_mask], self.df.loc[:,'dbase_balance'][base_mask],    c='green', linestyle='-', marker='o')
        ax3.set_title(f'Equivalent percentage changes')
        ax3.legend(loc="upper left")
        ax3.set_ylabel(f'{quote} balance percent change')
        ax5.set_ylabel(f'{pair.replace(quote, "")} balance percent change')
        ax3.tick_params(axis='y',  colors='blue')
        ax5.tick_params(axis='y',  colors='green')
        ax3.axhline(0, color='blue',  linestyle='--')
        ax5.axhline(0, color='green', linestyle='--')
        plt.subplots_adjust(hspace=10)
        plt.show()

        return


    def plot_price_and_indicators(self, **kwargs):

        fig = make_subplots(rows  = 2,
                            cols  = 1,
                            specs = [[{"secondary_y": True}], [{"secondary_y": True}]])

        # Candles
        candle = go.Candlestick(x     = self.df.index,
                                open  = self.df['open_h'],
                                close = self.df['close_h'],
                                high  = self.df['high_h'],
                                low   = self.df['low_h'],
                                name  = f'{self.base} price in {self.quote}')
        # Set line and fill colors of candles
        cs = candle
        cs.increasing.fillcolor  = 'white'      # '#3D9970'    # '#008000'
        cs.increasing.line.color = 'white'      # '#3D9970'    # '#008000'
        cs.decreasing.fillcolor  = 'black'      # '#FF4136'    # '#800000'
        cs.decreasing.line.color = 'black'      # '#FF4136'    # '#800000'
        fig.add_trace(candle, row=1, col=1)

        # Volume
        volume = go.Bar(x     = self.df.index,
                        y     = self.df['volume_h'],
                        # xaxis ="x",
                        # yaxis ="y2",
                        # width = 400000,
                        name  = "Volume",
                        )
        fig.add_trace(volume, row=1, col=1, secondary_y=True)

        # Buys and sells
        buys = go.Scatter(x      = self.df.index,
                          y      = self.df['buyprice_'+self.pair],
                          mode   = 'markers',
                          marker = dict(color='red', size=10, symbol='x-dot'),
                          name   = 'buys')
        fig.add_trace(buys,  row=1, col=1)
        sells = go.Scatter(x      = self.df.index,
                           y      = self.df['sellprice_'+self.pair],
                           mode   = 'markers',
                           marker = dict(color='green', size=10, symbol='x-dot'),
                           name   = 'sells')
        fig.add_trace(sells, row=1, col=1)

        # Add the indicators
        for indic_dict in self.indicators_dict.values():
            indic      = indic_dict['indic']
            indic_name = indic_dict['indic_name']
            ind = go.Scatter(x      = self.df.index,
                             y      = self.df[indic_name],
                             mode   = 'lines',
                             name   = indic_name,)
            fig.add_trace(ind, row=2 if 'rsi' in indic else 1, col=1)


        fig.update_layout({
            "title"     : f"{self.pair} - {self.strategy_name} Strategy - {self.timeframe} - {self.quote} profits: {round(self.quote_profits,1)}% (buy & hold: {round(self.buy_hold,1)}%). Stoploss: {self.stop_loss_pct}%.",
            "margin": {"t": 30, "b": 20},
            # "height": 800,
            "xaxis" : {
                "fixedrange"    : False,
                "showline"      : False,
                "zeroline"      : False,
                "showgrid"      : False,
                "showticklabels": True,
                "rangeslider"   : {"visible": False},
                "color"         : "#a3a7b0",
            },
            "yaxis" : {
                "fixedrange"    : False,
                "showline"      : False,
                "zeroline"      : False,
                "showgrid"      : False,
                "showticklabels": True,
                "ticks"         : "",
                "color"         : "#a3a7b0",
            },
            "yaxis2" : {
                "fixedrange"    : False,
                "showline"      : False,
                "zeroline"      : False,
                "showgrid"      : False,
                "showticklabels": True,
                "ticks"         : "",
                # "color"        : "#a3a7b0",
                "range"         : [0, self.df['volume_h'].max() * 10],
            },
            "legend" : {
                "font"          : dict(size=15, color="#a3a7b0"),
            },
            "plot_bgcolor"  : "#23272c",
            "paper_bgcolor" : "#23272c",

        })
        plot(fig)


    """ Helper methods """
    def find_high_of_last_red_candle(self, start_at:int):
        """ Returns the high of the last red candle.
            Used in the Gap and Go strategy."""
        is_red = False
        i = start_at
        while not is_red:
            if self.df.loc[self.df.index[i], 'close_h'] < self.df.loc[self.df.index[i], 'open_h']:
                is_red = True
            else:
                i-=1
        return self.df.loc[self.df.index[i], 'high_h']


    def find_low_of_last_green_candle(self, start_at:int):
        """ Returns the low of the last green candle.
            Used in the Gap and Go strategy."""
        is_green = False
        i = start_at
        while not is_green:
            if self.df.loc[self.df.index[i], 'close_h'] > self.df.loc[self.df.index[i], 'open_h']:
                is_green = True
            else:
                i-=1
        return self.df.loc[self.df.index[i], 'low_h']


class OptunaOptimization:

    def __init__(self, timeframe:str, quote:str, pair:str, strategy_name:str, **kwargs):
        self.timeframe = timeframe
        self.quote = quote
        self.pair = pair
        self.strategy_name = strategy_name
        self.kwargs = kwargs
        self.optimize_on = kwargs.get('optimize_on', 'profits')


    def objective(self, trial):
        """ Function to be maximized """

        length_1 = trial.suggest_int('length_1', 50, 200)
        # length_2 = trial.suggest_int('length_2', 30, 60)

        indic_2 = {}
        rsi_lower_threshold = None
        rsi_upper_threshold = None
        length_2 = 0

        # Suggest values of the hyperparameters using a trial object.
        indic_1 = dict(indicator='ssf' if self.strategy_name!='RSI' else 'rsi', length=length_1, close='close')
        if self.strategy_name=='Crossover':
            indic_2 = dict(indicator='ssf',    length=length_2*24, close='close')
        elif self.strategy_name=='MA_slope':
            indic_2 = dict(indicator='slope',  length=length_2*24, close=f'ssf_{length_1*24}')
        elif self.strategy_name=='Mean_Reversion_spread':
            indic_2	= dict(indicator='bbands', length=length_2*24, close='spread')
        elif self.strategy_name=='RSI':
            rsi_lower_threshold = trial.suggest_int('rsi_lower_threshold', 30, 70)
            # rsi_upper_threshold = trial.suggest_int('rsi_upper_threshold', 30, 70)

        stop_loss_pct = trial.suggest_int('stop_loss_pct', 2, 4)

        backtester = BackTesting(self.timeframe,
                                 quote   		      = self.quote,
                                 pair      		      = self.pair,
                                 strategy_name	      = self.strategy_name,				# Crossover, Mean_Reversion_simple, MA_slope
                                 starting_balances    = dict(quote=1, base=0),
                                 indic_1		      = indic_1,
                                 indic_2		      = indic_2,
                                 alloc_pct            = 100,
                                 plot                 = False,
                                 stop_loss_pct        = stop_loss_pct,
                                 rsi_lower_threshold  = rsi_lower_threshold,
                                 rsi_upper_threshold  = rsi_upper_threshold,
                                 display_progress_bar = False,
                                 print_output         = False,
                                 )
        quote_profits, metric, max_drawdown = backtester.backtest(**self.kwargs)

        if metric != 0:
            if self.optimize_on=='profits':
                return quote_profits
            elif self.optimize_on=='metric':
                return metric
            elif self.optimize_on=='profits_to_drawdown_ratio':
                return quote_profits/abs(max_drawdown)
            elif self.optimize_on=='multi_objective':
                return quote_profits, abs(max_drawdown)
            else:
                return None


    def run_optuna_optimization(self, n_trials:int, show_progress_bar:bool):
        # Create a study object and optimize the objective function.

        best_params, best_value = 0, 0

        if self.optimize_on == 'multi_objective':
            study = optuna.create_study(study_name=self.strategy_name, directions=["maximize", "minimize"])
            study.optimize(self.objective, n_trials=n_trials, show_progress_bar=show_progress_bar)
            trials = sorted(study.best_trials, key=lambda t: t.values)
            time.sleep(0.1) # aestetics
            for trial in trials:
                print("  Trial#{}".format(trial.number))
                print("    Values: Profit={}, max drawdown={}".format(trial.values[0], trial.values[1]))
                print("    Params: {}".format(trial.params))
        else:
            study = optuna.create_study(study_name=self.strategy_name, direction='maximize')
            study.optimize(self.objective, n_trials=n_trials, show_progress_bar=show_progress_bar)
            best_params = study.best_params
            best_value  = study.best_value
            print(f"\nBest parameters: {best_params}.\nBest value: {best_value}.")
            # plot(optuna.visualization.plot_optimization_history(study))
            # plot(optuna.visualization.plot_param_importances(study))

        return best_params, best_value



if __name__ == '__main__':

    # """ Run a single backtest """
    BackTesting(timeframe		  = '1h',
                quote    		  = 'ETH',
                pair      		  = 'ETHBTC',
                strategy_name     = 'RSI',									                            # Crossover, MA_slope, Mean_Reversion_simple, Mean_Reversion_spread, Gap_and_go,
                starting_balances = dict(quote=1, base=0),
                # indic_1			  = dict(indicator='ssf', length=14*24, close='close'),		        # Crossover	( best : 5*24, 40*24 )
                # indic_2			  = dict(indicator='ssf', length=31*24, close='close'),		        # Crossover
                indic_1			  = dict(indicator='rsi',    length=169, close='close'),		        # RSI
                # indic_2			  = dict(indicator='bbands', length=40*24, close=f'rsi_{1*24}'),    # RSI   5-55, 10-50
                rsi_lower_threshold = 54,		                                                        # RSI
                rsi_upper_threshold = 54,		                                                        # RSI
                # indic_1			  = dict(indicator='ssf',   length=100, close='close'),		        # MA_slope
                # indic_2			  = dict(indicator='slope', length=30,  close='ssf_100'),	        # MA_slope
                # indic_1			  = dict(indicator='ssf', length=500,  close='close'),		        # Mean_Reversion_simple
                # indic_1			  = dict(indicator='ssf',    length=80, close='close'),		        # Mean_Reversion_spread
                # indic_2			  = dict(indicator='bbands', length=80, close='spread'),	        # Mean_Reversion_spread
                alloc_pct         = 100,
                plot              = True,
                stop_loss_pct     = 3,    # 3
                trailing_stop     = False,
                display_progress_bar = True,
                print_output      = True,
                ).backtest(readjust_best_parameters=False)

    """ Run an Optuna Optimization """
    # OptunaOptimization(timeframe     = '1h',
    #                    quote         = 'ETH',
    #                    pair          = 'LTCETH',
    #                    strategy_name = 'RSI',
    #                    optimize_on   = 'profits_to_drawdown_ratio',                      # 'profits'| 'metric' | 'profits_to_drawdown_ratio' | 'multi_objective'
    #                    ).run_optuna_optimization(n_trials=150,
    #                                              show_progress_bar=True)


    # ETHBTC ---------------------------------------------------------------------------------------------------------------------------------------
    # 1h :     {'length_1': 169, 'rsi_lower_threshold': 54, 'stop_loss_pct': 3}. Best is trial 107 with value: 51.89.    (394.4%/-7.6%, sharpe=0.34)
    # 15 min : {'length_1': 309, 'rsi_lower_threshold': 56, 'stop_loss_pct': 3}. Best is trial 42 with value: 15.25.

    # LTCETH ---------------------------------------------------------------------------------------------------------------------------------------
