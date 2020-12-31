from Exchange     import Binance
from decimal  import Decimal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from empyrical import sortino_ratio, calmar_ratio, omega_ratio, sharpe_ratio
import pandas_ta as ta


class BackTesting:

	def __init__(self, timeframe):
		self.exchange  = Binance(filename='credentials.txt')
		self.timeframe = timeframe
		self.df        = pd.DataFrame()


	def prepare_df(self, quote:str, pair:str):

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

		min_ = int(len(self.df.index)*0.2)
		# max_ = None
		max_ = int(len(self.df.index)*0.3)
		self.df = self.df.iloc[min_:max_]

		# Drop all the non-necessary minute data : since we shifted, drop averythime at non hours indexes, where hours data is at NaN
		self.df.dropna(inplace=True)

		# print(self.df)
		print('Data is ready.')


	def backtest(self, quote:str, pair:str, starting_balances:dict, indic:str, length_fast:int, length_slow:int, alloc_pct:int, stop_loss_pct, plot:bool=False):

		self.prepare_df(quote=quote, pair=pair)

		fast_name = f'{indic}_{length_fast}'
		slow_name = f'{indic}_{length_slow}'

		# Compute the fast and slow averages
		getattr(self.df.ta, indic)(close=self.df.loc[:,pair+'_h'], length=length_fast, append=True, col_names=(fast_name,))
		getattr(self.df.ta, indic)(close=self.df.loc[:,pair+'_h'], length=length_slow, append=True, col_names=(slow_name,))
		# self.df.ta.bbands(close=self.df.loc[:,pair+'_h'], length=length_fast, append=True, col_names=('bband_l','bband_m','bband_u'))	# Used as a dynamic stop-loss. Marche moins bien.

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
		for i in tqdm(range(length_slow, len(hourly_close)-1)):      # tqdm : progress bar

			price 		 = self.df[pair+'_h'].iloc[i]
			fast         = self.df[fast_name].iloc[i]
			slow         = self.df[slow_name].iloc[i]
			fast_shifted = self.df[fast_name].iloc[i-1]
			slow_shifted = self.df[slow_name].iloc[i-1]
			slow_shifted2 = self.df[slow_name].iloc[i-2]
			self.df.loc[self.df.index[i],'signal'] = 'sell' if fast_shifted > slow_shifted and fast < slow else 'buy' if fast_shifted < slow_shifted and fast > slow else ''
			# self.df.loc[self.df.index[i],'signal'] = 'buy' if slow_shifted2 > slow_shifted and slow_shifted < slow else 'sell' if slow_shifted2 < slow_shifted and slow_shifted > slow else ''

			if status=='just bought': 	# ____________________________________________________________________________________________________________________________
				# Sell either by signal or stop-loss
				price_now    = Decimal(self.df[pair+'_h'].iloc[i])
				price_at_buy = Decimal(self.df.loc[:, 'buyprice_'+pair].dropna().iloc[-1])
				stop_loss_trigger = (price_now/price_at_buy-1)*100 < Decimal(-stop_loss_pct)
				increased_more_than_fees = (price_now/price_at_buy-1)*100 > 1

				if (self.df['signal'].iloc[i]=='sell' and increased_more_than_fees) or stop_loss_trigger:

					# To simulate the spread, we sell on the following minute candle.
					price_base_next_minute = Decimal(self.df[pair+'_m'].iloc[i+1])
					self.df.loc[self.df.index[i], 'sellprice_'+pair] = price_base_next_minute

					base_quantity_to_sell   = self.exchange.RoundToValidQuantity(pair=pair, quantity=base_balance)
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

			elif self.df['signal'].iloc[i]=='buy' and status=='just sold':	# _____________________________________________________________________________________

				# To simulate the spread, we buy on the following minute candle (it has been shifted already, so it's on the same index).
				price_base_this_minute = Decimal(self.df[pair+'_h'].iloc[i])
				price_base_next_minute = Decimal(self.df[pair+'_m'].iloc[i+1])
				self.df.loc[self.df.index[i], 'buyprice_'+pair] = price_base_next_minute

				base_quantity_to_buy   = self.exchange.RoundToValidQuantity(pair=pair, quantity=quote_balance/price_base_this_minute*alloc_pct/100)
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
		metric  = sharpe_ratio(cleaned, annualization=1)
		metric  = metric if abs(metric) != np.inf and not np.isnan(metric) else 0
		print(f'Sharp ratio : {round(metric,2)}')

		if plot:
			self.plotBotBacktest(quote	       = quote,
								 pair	   	   = pair,
								 indic         = indic,
								 length_fast   = length_fast,
								 length_slow   = length_slow,
								 alloc_pct     = alloc_pct,
								 trades        = trades,
								 stoploss      = stop_loss_pct,
								 metric		   = metric,
								 )

		return quote_profits, metric


	def plotBotBacktest(self, quote:str, pair:str, indic:str, length_fast:int, length_slow:float, alloc_pct:int, trades:dict, stoploss, metric):

		fast_name = f'{indic}_{length_slow}'
		slow_name = f'{indic}_{length_fast}'

		min_indice = -2000
		max_indice = None
		fig, ax1 = plt.subplots(figsize=(14,10))

		# First plot
		ax1.plot(self.df.index[min_indice:max_indice], self.df[pair+'_h'].iloc[min_indice:max_indice], color='black',  label=f"{pair.replace(quote, '')} price in {quote}")
		# Add the buys and sells
		ax1.scatter(self.df.index[min_indice:max_indice], self.df['buyprice_'+pair].iloc[min_indice:max_indice],  color='red',    marker='x', s=65, label='Buys')
		ax1.scatter(self.df.index[min_indice:max_indice], self.df['sellprice_'+pair].iloc[min_indice:max_indice], color='green',  marker='x', s=65, label='Sells')
		# Add the emas
		ax1.plot(self.df.index[min_indice:max_indice], self.df[fast_name].iloc[min_indice:max_indice], color='blue',   label=fast_name)
		ax1.plot(self.df.index[min_indice:max_indice], self.df[slow_name].iloc[min_indice:max_indice], color='yellow', label=slow_name)
		# Legend and tites
		ax1.set_title(f'CrossOver Strategy  -  {self.timeframe}    -    {indic} = ({length_fast}, {length_slow})\n\nPrices of {pair.replace(quote, "")} in {quote}')
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
		ax2.set_title(f'CrossOver Strategy  -  {self.timeframe}    -    Stop-loss={stoploss}%    -    {indic} = ({length_fast}, {length_slow})    -    alloc={alloc_pct}%    -    Trades : {trades["nb_trades"]}    -    Metric={round(metric,2)}\n\nQuantity of coins held')
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




if __name__ == '__main__':

	backtester = BackTesting('5m')
	# backtester.backtest(quote    		  = 'BTC',
	# 					pair      		  = 'ETHBTC',
	# 					starting_balances = dict(quote=1, base=0),
	# 					indic			  = 'ssf',
	# 					length_fast       = 5*12,		# 5*24
	# 					length_slow       = 60*12,		# 40*24
	# 					alloc_pct         = 100,
	# 					plot              = True,
	# 					stop_loss_pct     = 2,
	# 					)

	results                  = dict()
	results['length_fast']   = []
	results['length_slow']   = []
	results['stop_loss_pct'] = []
	results['quote_profits'] = []

	# Find the best lengths for this timeframe
	for length_fast_ in np.linspace(start=5, stop=20, num=4):
		for length_slow_ in np.linspace(start=20, stop=50, num=4):
			for stop_loss_pct_ in np.linspace(start=2, stop=4, num=3):
				quote_profits_, _ = backtester.backtest(quote   		  = 'BTC',
													    pair      		  = 'ETHBTC',
													    starting_balances = dict(quote=1, base=0),
													    indic			  = 'ssf',
													    length_fast       = int(length_fast_)*12,
													    length_slow       = int(length_slow_)*12,
													    alloc_pct         = 100,
													    plot              = False,
												   	    stop_loss_pct     = stop_loss_pct_,
												    	)

				print(f'length_fast, length_slow, stop_loss_pct = {length_fast_}, {length_slow_}, {stop_loss_pct_}')
				print(f'BTC profits on ETH = {round(quote_profits_, 2)}%')
				print('________________________')

				results['length_fast'].append(length_fast_)
				results['length_slow'].append(length_slow_)
				results['stop_loss_pct'].append(stop_loss_pct_)
				results['quote_profits'].append(quote_profits_)


	# ______________________________________________
	# Plot the results of the grid search
	df_ = pd.DataFrame({'length_fast'   : results['length_fast'],
					    'length_slow'   : results['length_slow'],
					    'stop_loss_pct' : results['stop_loss_pct'],
					    'quote_profits' : results['quote_profits']})

	import plotly.graph_objs as go
	fig = go.Figure()
	fig.add_trace(go.Scatter3d(x=df_.loc[:,'length_fast'], y=df_.loc[:,'length_slow'], z=df_.loc[:,'stop_loss_pct'],
							   mode='markers',
							   marker=dict(
										size       = 5,
										color      = df_.loc[:,'quote_profits'],      	# set color to an array/list of desired values
										colorscale = 'Viridis',                   		# choose a colorscale
										opacity    = 0.8,
										colorbar   = dict(thickness = 20,
														  title     = "BTC profits %"),
										)
							   )
				  )
	fig.update_layout(scene = dict(
								   xaxis_title='Length fast * 12',
								   yaxis_title='Length slow * 12',
								   zaxis_title='stop_loss_pct',
								   ),
					  title = "ETHBTC Simple Crossover - 1h - Sharpe ratio",
					  )
	fig.show()