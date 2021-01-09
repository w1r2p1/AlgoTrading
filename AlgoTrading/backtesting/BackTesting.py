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
		self.strategy_name		= strategy_name
		self.starting_balances 	= starting_balances
		self.alloc_pct 			= alloc_pct
		self.stop_loss_pct 		= stop_loss_pct
		self.plot 				= plot

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

		# Will be used to set the first trading point
		self.longest_indicator = max([self.indicators_dict[str(i)]['indic_length'] for i in self.indicators_dict.keys()])


	def prepare_df(self):

		# Get the dataframes from the csv files, keep only the time and close columns
		df_hrs_ = pd.read_csv(f'../historical_data/{self.quote}/{self.timeframe}/{self.pair}_{self.timeframe}', sep='\t').loc[:,['time', 'close']]
		df_min_ = pd.read_csv(f'../historical_data/{self.quote}/1m/{self.pair}_1m', sep='\t').loc[:,['time', 'close']]
		# Rename the close columns to the pair's name
		df_hrs_.columns = ['time', self.pair+'_h']
		df_min_.columns = ['time', self.pair+'_m']
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
		# self.df.loc[:,pair+'_m'] = self.df.loc[:,pair+'_m'].shift(-1)

		min_ = int(len(self.df.index)*0.3)
		max_ = int(len(self.df.index)*1)
		self.df = self.df.iloc[min_:max_]

		# Drop all the non-necessary minute data : since we shifted, drop averythime at non hours indexes, where hours data is at NaN

		# print(self.df)
		# print('Data is ready.')
		self.df.dropna(inplace=True)


	def compute_indicators(self):

		if self.strategy_name == 'Mean_Reversion_spread':
			self.df.loc[:,'spread'] = np.log(self.df.loc[:,self.pair+'_h'].pct_change() + 1)

		for indic_dict in self.indicators_dict.values():
			indic           = indic_dict['indic']
			indic_name      = (indic_dict['indic_name'],) if indic!='bbands' else ('bband_l', 'bband_m', 'bband_u')
			indic_length    = indic_dict['indic_length']
			indic_close_col = indic_dict['indic_close_col']

			close = self.df.loc[:,self.pair+'_h'] if indic_close_col=='close' else indic_close_col
			getattr(self.df.ta, indic)(close=close, length=indic_length, append=True, col_names=indic_name)


	def find_signal(self, i:int)->str:
		""" Compute the indicators and look for a signal.
			Each strategy is only a template that needs to be fine tuned.
		"""

		if self.strategy_name=='Crossover':
			fast         = self.df[self.indicators_dict['1']['indic_name']].iloc[i]
			slow         = self.df[self.indicators_dict['2']['indic_name']].iloc[i]
			fast_shifted = self.df[self.indicators_dict['1']['indic_name']].iloc[i-1]
			slow_shifted = self.df[self.indicators_dict['2']['indic_name']].iloc[i-1]
			return 'buy' if fast_shifted < slow_shifted and fast > slow else 'sell' if fast_shifted > slow_shifted and fast < slow else ''

		if self.strategy_name=='MA_slope':
			ma            = self.df[self.indicators_dict['1']['indic_name']].iloc[i]
			slope         = self.df[self.indicators_dict['2']['indic_name']].iloc[i]
			ma_shifted1   = self.df[self.indicators_dict['1']['indic_name']].iloc[i-1]
			slope_shifted = self.df[self.indicators_dict['2']['indic_name']].iloc[i-1]
			self.df.loc[self.df.index[i], 'signal'] = 'buy' if slope==0 and ma_shifted1>ma else 'sell' if slope==0 and ma_shifted1<ma else ''

		if self.strategy_name=='Mean_Reversion_simple':
			indic         = self.df[self.indicators_dict['1']['indic_name']].iloc[i]
			price_now     = Decimal(self.df[self.pair+'_h'].iloc[i])
			price_shifted = Decimal(self.df[self.pair+'_h'].iloc[i-1])
			self.df.loc[self.df.index[i], 'signal'] = 'buy' if price_now<price_shifted and price_now<indic else 'sell' if price_now>price_shifted and price_now>indic  else ''

		if self.strategy_name=='Mean_Reversion_spread':
			spread         = self.df['spread'].iloc[i]
			spread_shifted = self.df['spread'].iloc[i-1]
			bband_l		   = self.df['bband_l'].iloc[i]
			bband_u		   = self.df['bband_u'].iloc[i]
			self.df.loc[self.df.index[i],'signal'] = 'sell' if spread_shifted < spread and spread > bband_u else 'buy' if spread_shifted > spread and spread < bband_l else ''


	def backtest(self):

		# For code readability
		pair 				= self.pair
		starting_balances 	= self.starting_balances
		alloc_pct 			= self.alloc_pct
		stop_loss_pct 		= self.stop_loss_pct
		bot = dict(self.database.GetBot(pair=pair))

		self.prepare_df()

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
					  nb_los_trades = 0,
					  )

		# Go through all candlesticks
		hourly_close = self.df.loc[:,pair+'_h']
		for i in tqdm(range(self.longest_indicator, len(hourly_close)-1)):      # tqdm : progress bar
			# for i in range(length_slow, len(hourly_close)-1):

			# Look for a signal
			self.df.loc[self.df.index[i],'signal'] = self.find_signal(i=i)

			# price_above_indic = self.df[pair+'_h'].iloc[i] > self.df[indic_name].iloc[i]
			# price_under_indic = self.df[pair+'_h'].iloc[i] < self.df[indic_name].iloc[i]

			if status=='just bought': 	# ____________________________________________________________________________________________________________________________
				# Sell either by signal or stop-loss
				price_now    = Decimal(self.df[pair+'_h'].iloc[i])
				price_at_buy = Decimal(self.df.loc[:, 'buyprice_'+pair].dropna().iloc[-1])
				stop_loss_trigger = (price_now/price_at_buy-1)*100 < Decimal(-stop_loss_pct)
				increased_more_than_fees = (price_now/price_at_buy-1)*100 > 1

				# if (self.df['signal'].iloc[i]=='sell' and increased_more_than_fees and price_above_indic) or stop_loss_trigger:
				if (self.df['signal'].iloc[i]=='sell' and increased_more_than_fees) or stop_loss_trigger:

					# To simulate the spread, we sell on the following minute candle.
					price_base_next_minute = Decimal(self.df[pair+'_m'].iloc[i+1])
					self.df.loc[self.df.index[i], 'sellprice_'+pair] = price_base_next_minute

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

			elif self.df['signal'].iloc[i]=='buy' and status=='just sold': # and price_under_indic:	# _____________________________________________________________________________________

				# To simulate the spread, we buy on the following minute candle (it has been shifted already, so it's on the same index).
				price_base_this_minute = Decimal(self.df[pair+'_h'].iloc[i])
				price_base_next_minute = Decimal(self.df[pair+'_m'].iloc[i+1])
				self.df.loc[self.df.index[i], 'buyprice_'+pair] = price_base_next_minute

				base_quantity_to_buy   = self.helpers.RoundToValidQuantity(bot=bot, quantity=quote_balance/price_base_this_minute*alloc_pct/100)
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


		if trades['nb_trades'] > 1:
			# Compute profits and compare to buy and hold
			base_balance_initiale  = self.df.loc[:, 'base_balance'].dropna().iloc[0]
			base_balance_finale    = self.df.loc[:, 'base_balance'].dropna().iloc[-1]
			quote_balance_initiale = self.df.loc[:, 'quote_balance'].dropna().iloc[0]
			quote_balance_finale   = self.df.loc[:, 'quote_balance'].dropna().iloc[-1]
			price_base_at_first_quotevalue = self.df.loc[self.df['base_balance'] == base_balance_initiale, pair+'_h'].iloc[0]
			price_base_at_last_quotevalue  = self.df.loc[self.df['base_balance'] == base_balance_finale,   pair+'_h'].iloc[-1]
			quote_profits = (Decimal(quote_balance_finale) / quote_balance_initiale - 1)*100
			buy_hold_     = (price_base_at_last_quotevalue / price_base_at_first_quotevalue - 1)*100

			print(f'{self.quote} profits of {pair} = {round(quote_profits, 1)}%  (buy & hold = {round(buy_hold_, 1)}%)')
			print(f'Winning trades : {trades["nb_win_trades"]} ({int(trades["nb_win_trades"]/(trades["nb_win_trades"]+trades["nb_los_trades"])*100)}%)')
			print(f'Losing trades  : {trades["nb_los_trades"]} ({int((1-trades["nb_win_trades"]/(trades["nb_win_trades"]+trades["nb_los_trades"]))*100)}%)')

			# Remove 0s & NaNs and compute metric
			temp    = self.df.loc[:,'quote_balance'].astype(float)
			cleaned = temp[np.where(temp, True, False)].dropna().pct_change()
			metric  = sharpe_ratio(cleaned, annualization=1)
			metric  = metric if abs(metric) != np.inf and not np.isnan(metric) else 0
			# print(f'Sharp ratio : {round(metric,2)}')

			if self.plot:
				self.plot_backtest(trades=trades, metri=metric)

			return quote_profits, metric
		else:
			print("Strategy didn't buy or sell.")
			return 0, 0


	def plot_backtest(self, **kwargs):

		quote	      = self.quote
		pair	   	  = self.pair
		alloc_pct     = self.alloc_pct
		stoploss      = self.stop_loss_pct
		trades        = kwargs.get('trades', {})
		metric        = kwargs.get('metric', 0)

		print(self.df.columns)
		print(self.df)

		min_indice = -2000
		max_indice = None
		fig, ax1 = plt.subplots(figsize=(14,10))

		# First plot
		ax1.plot(self.df.index[min_indice:max_indice], self.df[pair+'_h'].iloc[min_indice:max_indice], color='black',  label=f"{pair.replace(quote, '')} price in {quote}")
		# Add the buys and sells
		ax1.scatter(self.df.index[min_indice:max_indice], self.df['buyprice_'+pair].iloc[min_indice:max_indice],  color='red',    marker='x', s=65, label='Buys')
		ax1.scatter(self.df.index[min_indice:max_indice], self.df['sellprice_'+pair].iloc[min_indice:max_indice], color='green',  marker='x', s=65, label='Sells')
		# Add the emas
		for indic_dict in self.indicators_dict.values():
			indic_name      = indic_dict['indic_name']
			ax1.plot(self.df.index[min_indice:max_indice], self.df[indic_name].iloc[min_indice:max_indice],   label=indic_name)
		# Legend and tites
		ax1.set_title(f'{self.strategy_name} Strategy  -  {self.timeframe}\n\nPrices of {pair.replace(quote, "")} in {quote}')
		ax1.legend(loc="upper left")
		ax1.set_ylabel(f'Price of {pair.replace(quote, "")} in {quote}')
		ax1.tick_params(axis='y',  colors='blue')
		ax1.grid(linestyle='--', axis='y')
		plt.show()


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



class MpGridSearch:

	def __init__(self, timeframe:str, pair:str, strategy_name:str):
		self.timeframe = timeframe
		self.pair = pair
		self.strategy_name = strategy_name
		self.results                  = dict()
		self.results['length_fast']   = []
		self.results['length_slow']   = []
		self.results['length']   	  = []
		self.results['length_slope']  = []
		self.results['length_bbands'] = []
		self.results['stop_loss_pct'] = []
		self.results['quote_profits'] = []

		if strategy_name=='Crossover':
			self.fast          = np.linspace(start=5,  stop=20,  num=4)
			self.slow          = np.linspace(start=30, stop=100, num=8)
			self.stop_loss_pct = np.linspace(start=2,  stop=5,   num=4)
			self.paramlist = list(itertools.product(self.fast, self.slow, self.stop_loss_pct))
		if strategy_name=='MA_slope':
			self.indic         = np.linspace(start=5,  stop=20,  num=4)
			self.slope         = np.linspace(start=30, stop=100, num=8)
			self.stop_loss_pct = np.linspace(start=2,  stop=5,   num=4)
			self.paramlist = list(itertools.product(self.indic, self.slope, self.stop_loss_pct))
		if strategy_name=='Mean_Reversion_simple':
			self.indic         = np.linspace(start=5,  stop=20,  num=4)
			self.stop_loss_pct = np.linspace(start=2,  stop=5,   num=4)
			self.paramlist = list(itertools.product(self.indic, self.stop_loss_pct))
		if strategy_name=='Mean_Reversion_spread':
			self.indic         = np.linspace(start=5,  stop=20,  num=4)
			self.bbands_length = np.linspace(start=5,  stop=20,  num=4)
			self.stop_loss_pct = np.linspace(start=2,  stop=5,   num=4)
			self.paramlist = list(itertools.product(self.indic, self.bbands_length, self.stop_loss_pct))

	def run_backtest(self, params:list):

		indic_1 = {}
		indic_2 = {}

		if self.strategy_name=='Crossover':
			indic_1 = dict(indicator='ssf',    length=int(params[0])*24, close='close')
			indic_2 = dict(indicator='ssf',    length=int(params[1])*24, close='close')
		elif self.strategy_name=='MA_slope':
			indic_1 = dict(indicator='ssf',    length=int(params[0]),    close='close')
			indic_2 = dict(indicator='slope',  length=int(params[1]),    close=f'ssf_{int(params[0])}')
		elif self.strategy_name=='Mean_Reversion_simple':
			indic_1	= dict(indicator='ssf',    length=int(params[0]),    close='close')
		elif self.strategy_name=='Mean_Reversion_spread':
			indic_1	= dict(indicator='ssf',    length=int(params[0]),    close='close')
			indic_2	= dict(indicator='bbands', length=int(params[1]),    close='spread')

		backtester = BackTesting(self.timeframe,
								 quote   		   = 'BTC',
								 pair      		   = 'ETHBTC',
								 strategy_name	   = self.strategy_name,				# Crossover, Mean_Reversion_simple, MA_slope
								 starting_balances = dict(quote=1, base=0),
								 indic_1		   = indic_1,
								 indic_2		   = indic_2,
								 alloc_pct         = 100,
								 plot              = False,
								 stop_loss_pct     = params[-1],
								 )
		quote_profits_, _ = backtester.backtest()

		# print(f'length_fast: {int(params[0])}*24 \t length_slow: {int(params[1])}*24 \t stop_loss_pct: {round(params[2],2)}% \t BTC profits on ETH: {round(quote_profits_,2)}%')
		print(f'{[params[i] for i in range(len(params))]}')

		if self.strategy_name=='Crossover':
			self.results['length_fast'].append(params[0])
			self.results['length_slow'].append(params[1])

		elif self.strategy_name=='MA_slope':
			self.results['length'].append(params[0])
			self.results['length_slope'].append(params[1])
		elif self.strategy_name=='Mean_Reversion_spread':
			self.results['length'].append(params[0])
			self.results['length_bbands'].append(params[1])

		self.results['stop_loss_pct'].append(params[-1])
		self.results['quote_profits'].append(quote_profits_)

		return self.results

	def run_grid_search(self):
		pool = Pool()
		res  = pool.map(self.run_backtest, self.paramlist)
		pool.close()
		pool.join()

		self.plot_gridsearch_result()

	def plot_gridsearch_result(self):
		""" Plot the results of the grid search """

		import plotly.graph_objs as go
		fig = go.Figure()

		if self.strategy_name=='Crossover':
			df_ = pd.DataFrame({'length_fast'   : self.results['length_fast'],
								'length_slow'   : self.results['length_slow'],
								'stop_loss_pct' : self.results['stop_loss_pct'],
								'quote_profits' : self.results['quote_profits']})

			fig.add_trace(go.Scatter3d(x=df_.loc[:,'length_fast'], y=df_.loc[:,'length_slow'], z=df_.loc[:,'stop_loss_pct'],
									   mode='markers',
									   marker=dict(
										   size       = 5,
										   color      = df_.loc[:,'quote_profits'],      	# set color to an array/list of desired values
										   colorscale = 'Viridis',                   		# choose a colorscale
										   opacity    = 0.8,
										   colorbar   = dict(thickness = 20,
															 title     = "BTC profits %"),
									   )))
			fig.update_layout(scene = dict(xaxis_title='Length fast',
										   yaxis_title='Length slow',
										   zaxis_title='stop_loss_pct',))

		if self.strategy_name=='Mean_Reversion_simple':
			df_ = pd.DataFrame({'length'        : self.results['length'],
								'stop_loss_pct' : self.results['stop_loss_pct'],
								'quote_profits' : self.results['quote_profits']})
			fig.add_trace(go.Scatter3d(x=df_.loc[:,'length'], y=df_.loc[:,'stop_loss_pct'], z=df_.loc[:,'quote_profits'],
									   mode='markers',
									   marker=dict(
										   size       = 5,
										   color      = 'blue',      	# set color to an array/list of desired values
									   )))
			fig.update_layout(scene = dict(xaxis_title='length',
										   yaxis_title='stop_loss_pct',
										   zaxis_title='quote_profits',))

		if self.strategy_name=='MA_slope':
			df_ = pd.DataFrame({'length'        : self.results['length'],
								'length_slope'  : self.results['length_slope'],
								'stop_loss_pct' : self.results['stop_loss_pct'],
								'quote_profits' : self.results['quote_profits']})
			fig.add_trace(go.Scatter3d(x=df_.loc[:,'length'], y=df_.loc[:,'length_slope'], z=df_.loc[:,'stop_loss_pct'],
									   mode='markers',
									   marker=dict(
										   size       = 5,
										   color      = df_.loc[:,'quote_profits'],      	# set color to an array/list of desired values
										   colorscale = 'Viridis',                   		# choose a colorscale
										   opacity    = 0.8,
										   colorbar   = dict(thickness = 20,
															 title     = "BTC profits %"),
									   )))
			fig.update_layout(scene = dict(xaxis_title='Length',
										   yaxis_title='Length slope',
										   zaxis_title='stop_loss_pct',))

		fig.update_layout(title = f"{self.pair} - {self.strategy_name} - {self.timeframe}",)
		fig.show()


if __name__ == '__main__':

	""" Run a single backtest """
	BackTesting(timeframe		  = '1h',
				quote    		  = 'BTC',
				pair      		  = 'ETHBTC',
				strategy_name     = 'Crossover',									# Crossover, MA_slope, Mean_Reversion_simple, Mean_Reversion_spread
				starting_balances = dict(quote=1, base=0),
				indic_1			  = dict(indicator='ssf', length=5*24,  close='close'),		# Crossover	( best : 5*24, 40*24 )
				indic_2			  = dict(indicator='ssf', length=40*24, close='close'),		# Crossover
				# indic_1			  = dict(indicator='ssf',   length=100, close='close'),		# MA_slope
				# indic_2			  = dict(indicator='slope', length=30,  close='ssf_100'),	# MA_slope
				# indic_1			  = dict(indicator='ssf', length=500,  close='close'),		# Mean_Reversion_simple
				# indic_1			  = dict(indicator='ssf',    length=80, close='close'),		# Mean_Reversion_spread
				# indic_2			  = dict(indicator='bbands', length=80, close='spread'),		# Mean_Reversion_spread
				alloc_pct         = 100,
				plot              = True,
				stop_loss_pct     = 2,
				).backtest()


	# results                  = dict()
	# results['length_fast']   = []
	# results['length_slow']   = []
	# results['stop_loss_pct'] = []
	# results['quote_profits'] = []
	#
	# # Find the best lengths for this timeframe
	# for length_fast_ in np.linspace(start=5, stop=20, num=4):
	# 	for length_slow_ in np.linspace(start=30, stop=100, num=8):
	# 		for stop_loss_pct_ in np.linspace(start=2, stop=5, num=4):
	# 			quote_profits_, _ = backtester.backtest(quote   		  = 'BTC',
	# 												    pair      		  = 'ETHBTC',
	# 												    starting_balances = dict(quote=1, base=0),
	# 												    indic			  = 'ssf',
	# 												    length_fast       = int(length_fast_)*24,
	# 												    length_slow       = int(length_slow_)*24,
	# 												    alloc_pct         = 100,
	# 												    plot              = False,
	# 											   	    stop_loss_pct     = stop_loss_pct_,
	# 											    	)
	#
	# 			print(f'length_fast, length_slow, stop_loss_pct = {length_fast_}, {length_slow_}, {stop_loss_pct_}')
	# 			print(f'BTC profits on BTC = {round(quote_profits_, 2)}%')
	# 			print('________________________')
	#
	# 			results['length_fast'].append(length_fast_)
	# 			results['length_slow'].append(length_slow_)
	# 			results['stop_loss_pct'].append(stop_loss_pct_)
	# 			results['quote_profits'].append(quote_profits_)
	#
	#
	# # ______________________________________________
	# # Plot the results of the grid search
	# df_ = pd.DataFrame({'length_fast'   : results['length_fast'],
	# 				    'length_slow'   : results['length_slow'],
	# 				    'stop_loss_pct' : results['stop_loss_pct'],
	# 				    'quote_profits' : results['quote_profits']})
	#
	# import plotly.graph_objs as go
	# fig = go.Figure()
	# fig.add_trace(go.Scatter3d(x=df_.loc[:,'length_fast'], y=df_.loc[:,'length_slow'], z=df_.loc[:,'stop_loss_pct'],
	# 						   mode='markers',
	# 						   marker=dict(
	# 									size       = 5,
	# 									color      = df_.loc[:,'quote_profits'],      	# set color to an array/list of desired values
	# 									colorscale = 'Viridis',                   		# choose a colorscale
	# 									opacity    = 0.8,
	# 									colorbar   = dict(thickness = 20,
	# 													  title     = "BTC profits %"),
	# 									)
	# 						   )
	# 			  )
	# fig.update_layout(scene = dict(
	# 							   xaxis_title='Length fast * 24',
	# 							   yaxis_title='Length slow * 24',
	# 							   zaxis_title='stop_loss_pct',
	# 							   ),
	# 				  title = "ETHBTC Simple Crossover - 1h - Sharpe ratio",
	# 				  )
	# fig.show()

	""" Run a grid search """
	# MpGridSearch(timeframe     = '1h',
	# 			 pair          = 'ETHBTC',
	# 			 strategy_name = 'MA_slope',									# Crossover, Mean_Reversion_simple, MA_slope
	# 			 ).run_grid_search()