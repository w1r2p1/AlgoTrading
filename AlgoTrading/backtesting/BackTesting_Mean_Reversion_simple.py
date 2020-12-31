import Strategies
from Exchange     import Binance

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


	def backtest(self, quote:str, pair:str, starting_balances:dict, indic:str, indic_length:int, alloc_pct:int, stop_loss_pct, plot:bool=False):
		""" Mean Reversion : If the spread is out of the bbands, signal.
			The indicator is a filter, for the buys and sells. 				"""

		self.prepare_df(quote=quote, pair=pair)

		indic_name = f'{indic}_{indic_length}'

		# Compute the spread, the bbands and the indic
		getattr(self.df.ta, indic)(close=self.df.loc[:,pair+'_h'], length=indic_length, append=True, col_names=(indic_name,))

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
		for i in tqdm(range(1, len(hourly_close)-1)):      # tqdm : progress bar

			price = self.df[pair+'_h'].iloc[i]
			price_shifted = self.df[pair+'_h'].iloc[i-1]
			indic_value = self.df[indic_name].iloc[i]
			self.df.loc[self.df.index[i],'signal'] = 'sell' if price < indic_value < price_shifted else 'buy' if price > indic_value > price_shifted else 0


			if status=='just bought':
				# Sell either by signal or stop-loss
				price_now    = Decimal(self.df[pair+'_h'].iloc[i])
				price_at_buy = Decimal(self.df.loc[:, 'buyprice_'+pair].dropna().iloc[-1])
				stop_loss_trigger = (price_now/price_at_buy-1)*100 < Decimal(-stop_loss_pct)
				increased_more_than_fees = (price_now/price_at_buy-1)*100 > 1

				if (self.df['signal'].iloc[i]=='sell' and increased_more_than_fees) or stop_loss_trigger:

					# To simulate the spread, we sell on the following minute candle.
					price_base_next_minute = Decimal(self.df[pair+'_m'].iloc[i+1])
					self.df.loc[self.df.index[i], 'sellprice_'+pair] = price_base_next_minute

					base_quantity_to_sell   = self.exchange.RoundToValidQuantity(pair=pair, quantity=base_balance*alloc_pct/100)
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

					status = 'just sold'

			elif self.df['signal'].iloc[i]=='buy' and status=='just sold':

				# To simulate the spread, we buy on the following minute candle (it has been shifted already, so it's on the same index).
				price_base_this_minute = Decimal(self.df[pair+'_h'].iloc[i])
				price_base_next_minute = Decimal(self.df[pair+'_m'].iloc[i+1])
				self.df.loc[self.df.index[i], 'buyprice_'+pair] = price_base_next_minute

				base_quantity_to_buy   = self.exchange.RoundToValidQuantity(pair=pair, quantity=quote_balance/price_base_this_minute*alloc_pct/100)
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
								 indic_length  = indic_length,
								 alloc_pct     = alloc_pct,
								 trades        = trades,
								 stoploss      = stop_loss_pct,
								 )

		return quote_profits, metric


	def plotBotBacktest(self, quote:str, pair:str, indic:str, indic_length:int, alloc_pct:int, trades:dict, stoploss):

		indic_name = f'{indic}_{indic_length}'

		# Pairs and spread ______________________________________________________________________________________________________________________
		min_indice = None
		max_indice = 1000
		fig, ax1 = plt.subplots(figsize=(14,12))

		# First plot
		ax1.plot(self.df.index[min_indice:max_indice], self.df[pair+'_h'].iloc[min_indice:max_indice], color='blue',  label=f"{pair.replace(quote, '')} price in {quote}")
		# Add the buys and sells
		ax1.scatter(self.df.index[min_indice:max_indice], self.df['buyprice_'+pair].iloc[min_indice:max_indice],  color='orange', marker='x', s=55, label='Buys')
		ax1.scatter(self.df.index[min_indice:max_indice], self.df['sellprice_'+pair].iloc[min_indice:max_indice], color='black',  marker='x', s=55, label='Sells')
		# Add the linreg
		ax1.plot(self.df.index[min_indice:max_indice], self.df[indic_name].iloc[min_indice:max_indice], color='green',  label=indic_name)
		# Legend and tites
		ax1.set_title(f'Simple Mean-Reversion  -  {self.timeframe}    -    indic:{indic_name}\n\nPrices of {pair.replace(quote, "")} in {quote}')
		ax1.legend(loc="upper left")
		ax1.set_ylabel(f'Price of {pair.replace(quote, "")} in {quote}')
		ax1.tick_params(axis='y',  colors='blue')
		plt.show()


		# Strategy evolution ______________________________________________________________________________________________________________________
		fig, ax1 = plt.subplots(figsize = (14,8))
		ax2 = ax1.twinx()
		quote_mask = np.isfinite(self.df.loc[:, 'quote_balance'].replace({0:np.nan}).astype(np.double))		# mask the NaNs for the plot, to allow to use plot with nan values
		base_mask  = np.isfinite(self.df.loc[:, 'base_balance'].replace({0:np.nan}).astype(np.double))
		ax1.plot(self.df.index[quote_mask], self.df.loc[:, 'quote_balance'][quote_mask], c='blue',  label=f"{quote} balance", linestyle='-', marker='o')
		ax1.plot([], [], 	 					         						         c='green', label=f"{pair.replace(quote, '')} balance", linestyle='-', marker='o')	# Empty plot just for the label
		ax2.plot(self.df.index[base_mask], self.df.loc[:, 'base_balance'][base_mask],    c='green', linestyle='-', marker='o')
		ax1.set_title(f'Simple Mean-Reversion  -  {self.timeframe}    -    Stop-loss={stoploss}%     -    alloc={alloc_pct}%    -    Trades : {trades["nb_trades"]}    -    indic:{indic_name}\n\nQuantity of coins held')
		ax1.legend(loc="upper left")
		ax1.set_ylabel(f'{quote} balance')
		ax2.set_ylabel(f'{pair.replace(quote, "")} balance')
		ax1.tick_params(axis='y',  colors='blue')
		ax2.tick_params(axis='y',  colors='green')
		ax1.axhline(self.df['quote_balance'].dropna().iloc[0], color='blue',  linestyle='--')
		ax2.axhline(self.df['base_balance'].dropna().iloc[0], color='green', linestyle='--')
		plt.subplots_adjust(hspace=10)
		plt.show()

		# Ajouter zero line pour les deux y axes du 1er plot, dans la couleur correspondante

		return




if __name__ == '__main__':

	backtester = BackTesting('1h')
	# backtester.backtest(quote    		  = 'BTC',
	# 					pair      		  = 'ETHBTC',
	# 					starting_balances = dict(quote=1, base=0),
	# 					indic         	  = 'ssf',
	# 					indic_length  	  = 40*24,
	# 					alloc_pct         = 100,		# How much of the coin balance we bet each time
	# 					plot              = True,
	# 					stop_loss_pct     = 2,
	# 					)

	results                  = dict()
	results['indic_length']  = []
	results['stop_loss_pct'] = []
	results['quote_profits'] = []

	# Find the best lengths for this timeframe
	for indic_length_ in np.linspace(start=1, stop=9, num=2):
		for stop_loss_pct_ in np.linspace(start=2, stop=4, num=2):
			quote_profits_, _ = backtester.backtest(quote   		  = 'BTC',
													pair      		  = 'ETHBTC',
													starting_balances = dict(quote=1, base=0),
													indic			  = 'ssf',
													indic_length      = indic_length_*24,
													alloc_pct         = 100,
													plot              = False,
													stop_loss_pct     = stop_loss_pct_,
													)

			print(f'indic_length, stop_loss_pct = {indic_length_}, {stop_loss_pct_}')
			print(f'BTC profits on ETH = {round(quote_profits_, 2)}%')
			print('________________________')

			results['indic_length'].append(indic_length_)
			results['stop_loss_pct'].append(stop_loss_pct_)
			results['quote_profits'].append(quote_profits_)

	# ______________________________________________
	# Plot the results of the grid search
	df_ = pd.DataFrame({'indic_length'  : results['indic_length'],
						'stop_loss_pct' : results['stop_loss_pct'],
						'quote_profits' : results['quote_profits']})

	import plotly.graph_objs as go
	fig = go.Figure()
	fig.add_trace(go.Scatter3d(x=df_.loc[:,'indic_length'], y=df_.loc[:,'stop_loss_pct'], z=df_.loc[:,'quote_profits'],
							   mode='markers',
							   marker=dict(
								   size       = 5,
								   color      = 'blue',      	# set color to an array/list of desired values
							   )
							   )
				  )
	fig.update_layout(scene = dict(
		xaxis_title='indic_length * 24',
		yaxis_title='stop_loss %',
		zaxis_title='quote_profits %',
	),
		title = "ETHBTC Simple Mean Reversion - 1h",
	)
	fig.show()