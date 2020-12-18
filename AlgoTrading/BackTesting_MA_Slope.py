import Strategies
from Exchange import Binance

from decimal  import Decimal, getcontext
import plotly.graph_objs as go
import pandas as pd
from plotly.subplots import make_subplots
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as tsa
import statsmodels.api as sm
from tqdm import tqdm
from empyrical import sortino_ratio, calmar_ratio, omega_ratio, sharpe_ratio
import pandas_ta as ta
import math
from supersmoother import SuperSmoother, LinearSmoother

class BackTesting:

	def __init__(self, timeframe):
		self.exchange  = Binance(filename='credentials.txt')
		self.timeframe = timeframe
		self.df        = pd.DataFrame()


	def prepare_df(self, quote:str, pair:str):

		min_ = 0
		# min_ = 6000
		max_ = None
		# max_ = 1700

		# Get the dataframes from the csv files, keep only the time and close columns
		df_hrs_ = pd.read_csv(f'historical_data/{quote}/{self.timeframe}/{pair}_{self.timeframe}', sep='\t').loc[:,['time', 'close']]
		df_min_ = pd.read_csv(f'historical_data/{quote}/1m/{pair}_1m', sep='\t').loc[:,['time', 'close']]
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

		self.df = self.df.iloc[min_:max_]

		# Drop all the non-necessary minute data : since we shifted, drop averythime at non hours indexes, where hours data is at NaN
		self.df.dropna(inplace=True)

		print('Data is ready.')


	def backtest(self, quote:str, pair:str, starting_balances:dict, indic:str, length_ma:int, length_slope:int, alloc_pct:int, stop_loss_pct, plot:bool=False):

		self.prepare_df(quote=quote, pair=pair)

		ma_name    = f'{indic}_{length_ma}'
		slope_name = f'slope{length_slope}_of_{indic}{length_ma}'

		# Compute the ma and its slope
		getattr(self.df.ta, indic)(close=self.df.loc[:,pair+'_h'], length=length_ma, append=True, col_names=(ma_name,))
		# getattr(self.df.ta, indic)(close=self.df.loc[:,pair+'_h'], length=length_ma*2, append=True, col_names=(ma_name+'*X',))
		self.df.ta.slope(close=ma_name, length=length_slope, append=True, col_names=(slope_name,))
		# self.df.ta.slope(close=slope_name, length=length_slope, append=True, col_names=('slope_of_' + slope_name,), fillna=np.nan)

		# self.df = self.df.iloc[length_ma*2:]

		# Set the first point
		quote_balance = Decimal(starting_balances['quote'])
		base_balance  = Decimal(starting_balances['base'])
		self.df.loc[self.df.index[0], 'quote_balance']   = quote_balance
		self.df.loc[self.df.index[0], 'base_balance']    = base_balance
		status = 'just sold'		# look to buy first

		# Initialize variables
		trades = dict(nb_trades     = 0,
					  nb_win_trades = 0,
					  nb_los_trades = 0,
					  )

		# Go through all candlesticks
		hourly_close = self.df.loc[:,pair+'_h']
		for i in tqdm(range(max(length_ma, length_slope), len(hourly_close)-1)):      # tqdm : progress bar

			ma    = self.df[ma_name].iloc[i]
			slope = self.df[slope_name].iloc[i]
			ma_shifted1  = self.df[ma_name].iloc[i-1]
			ma_shifted2  = self.df[ma_name].iloc[i-2]
			slope_shifted = self.df[slope_name].iloc[i-1]
			slope_of_slope = self.df['slope_of_' + slope_name].iloc[i]
			# Compute the unfiltered signals
			self.df.loc[self.df.index[i], 'signal'] = 'buy' if slope > 0 and slope_of_slope > 0 else 'sell' if slope < 0 and slope_of_slope < 0 else ''
			# self.df.loc[self.df.index[i], 'signal'] = 'buy' if ma_shifted2 > ma_shifted1 and ma_shifted1 < ma else 'sell' if ma_shifted2 < ma_shifted1 and ma_shifted1 > ma else ''
			price_now = Decimal(self.df[pair+'_h'].iloc[i])

			if status=='just bought': 	# ____________________________________________________________________________________________________________________________
				# Sell either by signal or stop-loss
				price_at_buy = Decimal(self.df.loc[:, 'buyprice_'+pair].dropna().iloc[-1])
				stop_loss_trigger = (price_now/price_at_buy-1)*100 < Decimal(-stop_loss_pct)
				# increased_more_than_fees = (price_now/price_at_buy-1)*100 > 2*Decimal(0.075)
				increased_more_than_fees = (price_now/price_at_buy-1)*100 > 1

				if (self.df['signal'].iloc[i]=='sell' and increased_more_than_fees and price_now>ma) or stop_loss_trigger:
					# if (self.df['signal'].iloc[i]=='sell' and increased_more_than_fees) or stop_loss_trigger:

					# To simulate the spread, we sell on the following minute candle.
					price_base_next_minute = Decimal(self.df[pair+'_m'].iloc[i+1])
					self.df.loc[self.df.index[i], 'sellprice_'+pair] = price_base_next_minute

					base_quantity_to_sell   = self.exchange.RoundToValidQuantity(pair=pair, quantity=base_balance*alloc_pct/100)
					# print(f'Base balance = {base_balance}, Sold {base_quantity_to_sell} {pair.replace(quote, "")}')
					quote_quantity_sell     = base_quantity_to_sell*price_base_next_minute
					fee_in_quote_sell       = quote_quantity_sell*Decimal(0.075)/Decimal(100)
					received_quote_quantity = quote_quantity_sell - fee_in_quote_sell						# What we get in quote from the sell

					base_balance  -= base_quantity_to_sell
					quote_balance += received_quote_quantity
					balance_quote_previous = self.df.loc[:, 'quote_balance'].dropna().iloc[-1]

					self.df.loc[self.df.index[i], 'quote_balance'] = quote_balance

					trades['nb_trades'] += 1
					if quote_balance < balance_quote_previous:
						trades['nb_los_trades'] += 1
					else:
						trades['nb_win_trades'] += 1

					self.df.loc[self.df.index[i], 'fees'] = fee_in_quote_sell
					self.df.loc[self.df.index[i], 'sells_on_slope'] = self.df.loc[self.df.index[i], slope_name]

					status = 'just sold'

			elif (self.df['signal'].iloc[i]=='buy' and status=='just sold') and price_now<ma:	# _____________________________________________________________________________________
				# elif self.df['signal'].iloc[i]=='buy' and status=='just sold':	# _____________________________________________________________________________________

				# To simulate the spread, we buy on the following minute candle (it has been shifted already, so it's on the same index).
				price_base_this_minute = Decimal(self.df[pair+'_h'].iloc[i])
				price_base_next_minute = Decimal(self.df[pair+'_m'].iloc[i+1])
				self.df.loc[self.df.index[i], 'buyprice_'+pair] = price_base_next_minute

				base_quantity_to_buy   = self.exchange.RoundToValidQuantity(pair=pair, quantity=quote_balance/price_base_this_minute*alloc_pct/100)
				# print(f'Base balance = {base_balance}, Bought {base_quantity_to_buy} {pair.replace(quote, "")}')
				quote_quantity_buy     = base_quantity_to_buy*price_base_next_minute
				fee_in_base_buy        = base_quantity_to_buy*Decimal(0.075)/Decimal(100)
				received_base_quantity = base_quantity_to_buy - fee_in_base_buy						# What we get in base from the buy

				base_balance  += received_base_quantity
				quote_balance -= quote_quantity_buy

				base_balance_previous = self.df.loc[:, 'base_balance'].dropna().iloc[-1]

				self.df.loc[self.df.index[i], 'base_balance'] = base_balance

				trades['nb_trades'] += 1
				if base_balance < base_balance_previous:
					trades["nb_los_trades"] += 1
				else:
					trades['nb_win_trades'] += 1

				self.df.loc[self.df.index[i], 'fees'] = fee_in_base_buy*price_base_next_minute
				self.df.loc[self.df.index[i], 'buys_on_slope'] = self.df.loc[self.df.index[i], slope_name]


				status = 'just bought'

			else:
				# Do nothing
				self.df.loc[self.df.index[i], 'fees'] = Decimal(0)


		# Compute profits and compare to buy and hold
		base_balance_initiale  = self.df.loc[:, 'base_balance'].dropna().iloc[0]
		base_balance_finale    = self.df.loc[:, 'base_balance'].dropna().iloc[-1]
		quote_balance_initiale = self.df.loc[:, 'quote_balance'].dropna().iloc[0]
		quote_balance_finale   = self.df.loc[:, 'quote_balance'].dropna().iloc[-1]
		price_base_at_first_quotevalue = self.df.loc[self.df['base_balance'] == base_balance_initiale, pair+'_h'].iloc[0]
		price_base_at_last_quotevalue  = self.df.loc[self.df['base_balance'] == base_balance_finale,   pair+'_h'].iloc[0]
		quote_profits = (Decimal(quote_balance_finale) / quote_balance_initiale - 1)*100
		buy_hold_     = (price_base_at_last_quotevalue / price_base_at_first_quotevalue - 1)*100

		print(f'{quote} profits of {pair} = {round(quote_profits, 1)}%  (buy & hold = {round(buy_hold_, 1)}%)')
		print(f'Winning trades : {trades["nb_win_trades"]} ({int(trades["nb_win_trades"]/(trades["nb_win_trades"]+trades["nb_los_trades"])*100)}%)')
		print(f'Losing trades  : {trades["nb_los_trades"]} ({int((1-trades["nb_win_trades"]/(trades["nb_win_trades"]+trades["nb_los_trades"]))*100)}%)')

		# Remove 0s & NaNs and compute metric
		temp    = self.df.loc[:,'quote_balance'].astype(float)
		cleaned = temp[np.where(temp, True, False)].dropna().pct_change()
		metric  = omega_ratio(cleaned, annualization=1)
		metric  = metric if abs(metric) != np.inf and not np.isnan(metric) else 0

		if plot:
			self.plotBotBacktest(quote	       = quote,
								 pair	   	   = pair,
								 indic         = indic,
								 length_ma     = length_ma,
								 length_slope  = length_slope,
								 alloc_pct     = alloc_pct,
								 trades        = trades,
								 stoploss      = stop_loss_pct,
								 )

		return quote_profits, metric


	def plotBotBacktest(self, quote:str, pair:str, indic:str, length_ma:int, length_slope:float, alloc_pct:int, trades:dict, stoploss):

		ma_name    = f'{indic}_{length_ma}'
		slope_name = f'slope{length_slope}_of_{indic}{length_ma}'

		# with pd.option_context('display.max_rows', 20, 'display.max_columns', None):
		# 	print(self.df)

		# Pairs and spread ______________________________________________________________________________________________________________________
		min_indice = 2000
		max_indice = 2700
		fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14,12))

		# First plot
		ax1.plot(self.df.index[min_indice:max_indice], self.df[pair+'_h'].iloc[min_indice:max_indice], color='black',  label=f"{pair.replace(quote, '')} price in {quote}")
		# Add the buys and sells
		ax1.scatter(self.df.index[min_indice:max_indice], self.df['buyprice_'+pair].iloc[min_indice:max_indice],  color='red',    marker='x', s=65, label='Buys')
		ax1.scatter(self.df.index[min_indice:max_indice], self.df['sellprice_'+pair].iloc[min_indice:max_indice], color='green',  marker='x', s=65, label='Sells')
		# Add the ma
		ax1.plot(self.df.index[min_indice:max_indice], self.df[ma_name].iloc[min_indice:max_indice], color='blue',   label=ma_name.replace('_', ' '))
		# ax1.plot(self.df.index[min_indice:max_indice], self.df[ma_name+'*X'].iloc[min_indice:max_indice], color='yellow',   label=ma_name.replace('_', ' ')+'*X')
		# Legend and tites
		ax1.set_title(f'Slope Strategy  -  {self.timeframe}    -    ({indic}, slope) = ({length_ma}, {length_slope})\n\nPrices of {pair.replace(quote, "")} in {quote}')
		ax1.legend(loc="upper left")
		ax1.set_ylabel(f'Price of {pair.replace(quote, "")} in {quote}')
		ax1.tick_params(axis='y',  colors='blue')
		ax1.grid(linestyle='--', axis='y')

		# Second plot
		ax2.plot(self.df.index[min_indice:max_indice], self.df[slope_name].iloc[min_indice:max_indice],          color='blue', label=slope_name.replace('_', ' '))
		ax2.scatter(self.df.index[min_indice:max_indice], self.df['buys_on_slope'].iloc[min_indice:max_indice],  color='red',   marker='x',  label="Buys")
		ax2.scatter(self.df.index[min_indice:max_indice], self.df['sells_on_slope'].iloc[min_indice:max_indice], color='green', marker='x',  label="Sells")
		ax2.set_title(slope_name.replace('_', ' '))
		ax2.set_xlabel('Date')
		ax2.set_ylabel(slope_name.replace('_', ' '))
		ax2.legend(loc="upper left")
		ax2.grid(linestyle='--', axis='y')
		plt.subplots_adjust(hspace = 10)
		plt.show()

		# Strategy evolution ______________________________________________________________________________________________________________________
		fig, ax1 = plt.subplots(figsize = (14,10))
		ax2 = ax1.twinx()
		ax1.scatter(self.df.index, self.df.loc[:, 'quote_balance'].replace({0:np.nan}), color='blue',  label=f"{quote} balance", s=15)
		ax1.scatter([], [], 	 					         						    color='green', label=f"{pair.replace(quote, '')} balance", s=15)	# Empty plot just for the label
		ax2.scatter(self.df.index, self.df.loc[:, 'base_balance'].replace({0:np.nan}),  color='green', s=15)
		ax1.set_title(f'Slope Strategy  -  {self.timeframe}    -    Stop-loss={stoploss}%    -     ({indic}, slope) = ({length_ma}, {length_slope})    -    alloc={alloc_pct}%    -    Trades : {trades["nb_trades"]}\n\nQuantity of coins held')
		ax1.legend(loc="upper left")
		ax1.set_ylabel(f'{quote} balance')
		ax2.set_ylabel(f'{pair.replace(quote, "")} balance')
		ax1.tick_params(axis='y',  colors='blue')
		ax2.tick_params(axis='y',  colors='green')
		ax1.axhline(1, color='blue',  linestyle='--')
		ax2.axhline(1, color='green', linestyle='--')
		plt.subplots_adjust(hspace=10)
		plt.show()

		return




if __name__ == '__main__':

	backtester = BackTesting('4h')
	backtester.backtest(quote    		  = 'BTC',
						pair      		  = 'ETHBTC',
						starting_balances = dict(quote=1, base=1),
						indic			  = 'wma',
						length_ma         = 40,
						length_slope      = 5,
						alloc_pct         = 100,
						plot              = True,
						stop_loss_pct     = 5,
						)

	# backtester.test_sideway_stationnarity(quote     = 'BTC',
	# 									  pair      = 'ETHBTC',
	# 									  length    = 10,
	# 									  )

	# results           = dict()
	# results['length'] = []
	# results['std']    = []
	# results['metric'] = []
	# results['quote_profits'] = []
	#
	# # Find the best length and std for this timeframe
	# for length_ in np.linspace(start=5, stop=80, num=5):
	# 	for std_ in np.linspace(start=1, stop=2.5, num=5):
	#
	# 		quote_profits_, metric_ = backtester.backtest(quote   = 'BTC',
	# 												      pair     = 'ETHBTC',
	# 												      starting_balances = dict(quote=1,base=1),
	# 												      length    = length_,
	# 												      std       = std_,
	# 												      alloc_pct = 100,
	# 													  stop_loss_pct=2
	# 													  )
	#
	# 		print(f'length, std = {length_}, {std_}')
	# 		print(f'metric = ', metric_)
	# 		print(f'BTC profits on ETH = {round(quote_profits_, 2)}%')
	# 		print('________________________')
	#
	# 		results['length'].append(length_)
	# 		results['std'].append(std_)
	# 		results['metric'].append(metric_)
	# 		results['quote_profits'].append(quote_profits_)
	#
	#
	# # ______________________________________________
	# # Plot the results of the grid search
	# df_ = pd.DataFrame({'length': results['length'],
	# 				   'std'    : results['std'],
	# 				   'metric' : results['metric'],
	# 				   'quote_profits': results['quote_profits']})
	#
	# import plotly.graph_objs as go
	# fig = go.Figure()
	# fig.add_trace(go.Scatter3d(x=df_.loc[:,'length'], y=df_.loc[:,'std'], z=df_.loc[:,'metric'],
	# 						   mode='markers',
	# 						   marker=dict(
	# 									size       = 5,
	# 									color      = df_.loc[:,'quote_profits'],      # set color to an array/list of desired values
	# 									colorscale = 'Viridis',                             # choose a colorscale
	# 									opacity    = 0.8,
	# 									colorbar   = dict(thickness = 20,
	# 													  title     = "BTC profits %"),
	# 									)
	# 						   )
	# 			  )
	# fig.update_layout(scene = dict(
	# 							  xaxis_title='Length',
	# 							  yaxis_title='STD',
	# 							  zaxis_title='Metric',
	# 							  ),
	# 				  title = "ETHBTC Mean-Reversion - 2h - Omega ratio",
	# 				 )
	# fig.show()