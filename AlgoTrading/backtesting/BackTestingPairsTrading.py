from Exchange import Binance

from decimal  import Decimal, getcontext
import plotly.graph_objs as go
import pandas as pd
from plotly.subplots import make_subplots
import numpy as np
import pandas_ta as ta
from functools import reduce
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from empyrical import sortino_ratio, calmar_ratio, omega_ratio, sharpe_ratio
import optuna
from scipy.interpolate import griddata
from matplotlib import cm
import statsmodels.tsa.stattools as tsa
import statsmodels.api as sm
from mlfinlab.optimal_mean_reversion import OrnsteinUhlenbeck
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

# # example of secondary Y in matplotlib
# fig, ax1 = plt.subplots(figsize = (16,12))
# ax2 = ax1.twinx()
# ax1.plot(finaldf.index[:100], finaldf['spread'].iloc[:100], 'r-')
# ax2.scatter(finaldf.index[:100], finaldf['short_signal_on_spread'].iloc[:100])
# ax2.scatter(finaldf.index[:100], finaldf['long_signal_on_spread'].iloc[:100])
# plt.show()

class BackTestingPairsTrading:
	""" Class used in 'BotDeMoi' to backtest a strategy. """

	def __init__(self, timeframe):
		self.exchange  = Binance(filename='../assets/credentials.txt')
		self.timeframe = timeframe
		self.df        = pd.DataFrame()


	def prepare_df(self, quote:str, pairs:list):

		min_ = 0
		# min_ = 6000
		max_ = None
		# max_ = 10000

		# Get the dataframes from the csv files, keep only the time and close columns
		df_pair0_hrs = pd.read_csv(f'../historical_data/{quote}/{self.timeframe}/{pairs[0]}_{self.timeframe}', sep='\t').loc[:,['time', 'close']]
		df_pair1_hrs = pd.read_csv(f'../historical_data/{quote}/{self.timeframe}/{pairs[1]}_{self.timeframe}', sep='\t').loc[:,['time', 'close']]
		df_pair0_min = pd.read_csv(f'../historical_data/{quote}/1m/{pairs[0]}_1m', sep='\t').loc[:,['time', 'close']]
		df_pair1_min = pd.read_csv(f'../historical_data/{quote}/1m/{pairs[1]}_1m', sep='\t').loc[:,['time', 'close']]
		# Rename the close columns to the pair's name
		df_pair0_hrs.columns = ['time', pairs[0]+'_h']
		df_pair1_hrs.columns = ['time', pairs[1]+'_h']
		df_pair0_min.columns = ['time', pairs[0]+'_m']
		df_pair1_min.columns = ['time', pairs[1]+'_m']
		# Set indexes
		df_pair0_hrs.set_index('time', inplace=True)
		df_pair1_hrs.set_index('time', inplace=True)
		df_pair0_min.set_index('time', inplace=True)
		df_pair1_min.set_index('time', inplace=True)

		# Test : if data of 2 timeframes match
		# print(df_pair0_hrs.loc['2017-12-28 12:00:00.000'])
		# print(df_pair0_min.loc['2017-12-28 12:00:00'])

		# Remove duplicated lines in the historical data if present
		df0_hrs = df_pair0_hrs.loc[~df_pair0_hrs.index.duplicated(keep='first')]
		df1_hrs = df_pair1_hrs.loc[~df_pair1_hrs.index.duplicated(keep='first')]
		df0_min = df_pair0_min.loc[~df_pair0_min.index.duplicated(keep='first')]
		df1_min = df_pair1_min.loc[~df_pair1_min.index.duplicated(keep='first')]

		# Reformat datetime index, Binance's data is messy
		df0_hrs.index = pd.to_datetime(df0_hrs.index, format='%Y-%m-%d %H:%M:%S.%f')
		df1_hrs.index = pd.to_datetime(df1_hrs.index, format='%Y-%m-%d %H:%M:%S.%f')
		df0_min.index = pd.to_datetime(df0_min.index, format='%Y-%m-%d %H:%M:%S.%f')
		df1_min.index = pd.to_datetime(df1_min.index, format='%Y-%m-%d %H:%M:%S.%f')

		# Merge in a single dataframe that has the time as index
		df_hours = pd.merge(df0_hrs, df1_hrs, how='outer', left_index=True, right_index=True)
		df_min   = pd.merge(df0_min, df1_min, how='outer', left_index=True, right_index=True)

		# If the two coins don't have enough overlapping, skip the pair.
		if len(df_hours.dropna()) < 1000:
			print(f'{pairs} - Less than 1000 ({len(df_hours)}) matching indexes, skipping the pair.')
			return

		# Create a large dataframe containing all the data
		self.df = pd.concat([df_hours, df_min], axis='columns').iloc[min_:max_]

		# # To simplify the code, shift the next minute data 1 place backwards so that the indice of the next minute candle matches the hours' one that gives the signal.
		# self.df.loc[:,pairs[0]+'_m'] = self.df.loc[:,pairs[0]+'_m'].shift(-1)
		# self.df.loc[:,pairs[1]+'_m'] = self.df.loc[:,pairs[1]+'_m'].shift(-1)

		# Drop all the non-necessary minute data : since we shifted, drop averythime at non hours indexes, where hours data is at NaN
		self.df.dropna(inplace=True)

		self.df = self.df.iloc[min_:max_]


	def backtest(self, quote:str, pairs:list, starting_balances:dict, length:int, std, alloc_pct:int, plot:bool=False):

		self.prepare_df(quote=quote, pairs=pairs)

		method = 'log_spread'
		beta = 1

		if method=='log_spread':
			# Compute the spread
			self.df.loc[:,'spread'] = np.log(self.df.loc[:,pairs[0]+'_h'].pct_change()+1) - np.log(self.df.loc[:,pairs[1]+'_h'].pct_change()+1)
			# self.df.loc[:,'spread'] = self.df.loc[:,pairs[0]+'_h'].pct_change() - self.df.loc[:,pairs[1]+'_h'].pct_change()
			self.df.ta.bbands(close=self.df.loc[:,'spread'], length=length, std=std, append=True, col_names=('bband_l', 'bband_m', 'bband_u'))

		elif method=='Regression':
			# run Odinary Least Squares regression to find hedge ratio and then create spread series
			# est    = sm.OLS(self.df.loc[:,pairs[0]+'_h'], self.df.loc[:,pairs[1]+'_h'], intercept=0)
			est    = sm.OLS(self.df.loc[:,pairs[0]+'_h'], self.df.loc[:,pairs[1]+'_h'], intercept=0)
			result = est.fit()
			# print(result.summary())
			beta   = result.params[0]		# Edge Ratio
			print('beta = ', beta)
			# Spread
			self.df.loc[:,'spread'] = self.df.loc[:,pairs[0]+'_h'] - beta*self.df.loc[:,pairs[1]+'_h']

			# Half life of mean reversion. Run OLS regression on spread series and lagged version of itself
			def get_halflife(s):
				# To determine the halflife, regress y(t) − y(t − 1) against y(t − 1)
				s_lag = s.shift(1)
				s_lag.iloc[0] = s_lag.iloc[1]

				s_ret = s - s_lag
				s_ret.iloc[0] = s_ret.iloc[1]

				s_lag2 = sm.add_constant(s_lag)

				model = sm.OLS(s_ret,s_lag2)
				res_   = model.fit()

				# print(res_.summary())

				halflife_ = int(-np.log(2) / res_.params[1])
				# halflife_ = int(-np.log(2) / np.log(np.abs(res_.params[1])))
				# halflife_ = np.log(0.5)/np.log(np.abs(res_.params[1]))
				return halflife_

			halflife = get_halflife(self.df.loc[:,'spread'])
			# halflife = 50
			print('Half-life = ', halflife)
			# Zscore of the spread
			self.df.ta.zscore(close=self.df.loc[:,'spread'], length=halflife, std=1, append=True, col_names=('zScore',))

		# Run Augmented Dickey-Fuller test to check if the time-series is stationnary
		result = tsa.adfuller(self.df['spread'].dropna(), autolag='AIC')
		# print('Spread : p-value : %f' % result[1])                                           # if p < 0.05, the time-series is stationnary

		# Set the first point
		balance_quote = Decimal(starting_balances['quote'])
		# Set equal quote values for the two coins
		balance_pair0 = Decimal(starting_balances['quote_for_pair0'])/Decimal(self.df.loc[:,pairs[0]+'_h'][0])
		balance_pair1 = Decimal(starting_balances['quote_for_pair1'])/Decimal(self.df.loc[:,pairs[1]+'_h'][0])
		print(balance_pair0, balance_pair1)
		self.df.loc[self.df.index[0], 'balance_quote']    = balance_quote
		self.df.loc[self.df.index[0], 'balance_pair0']    = balance_pair0
		self.df.loc[self.df.index[0], 'balance_pair1']    = balance_pair1
		self.df.loc[self.df.index[0], 'quotevalue_pair0'] = balance_pair1*Decimal(self.df[pairs[0]+'_h'].iloc[0])
		self.df.loc[self.df.index[0], 'quotevalue_pair1'] = balance_pair1*Decimal(self.df[pairs[1]+'_h'].iloc[0])
		status = ''

		# Initialize variables
		received_quantity_pair0 = 0
		received_quantity_pair1 = 0
		fee_in_pair0_buy        = 0
		fee_in_pair1_buy        = 0
		trades = dict(nb_trades     = 0,
					  nb_win_trades = 0,
					  nb_los_trades = 0,
					  )

		""" IMPLEMENTER STOP-LOSS """

		# Go through all candlesticks
		hourly_close = self.df.loc[:,pairs[0]+'_h']
		for i in tqdm(range(1, len(hourly_close)-1)):      # tqdm : progress bar

			# Work on the hours timeframe, so don't consider minutes != 00
			# if str(self.df.index[i])[14:16]=='00':

			spread         = self.df['spread'].iloc[i]
			spread_shifted = self.df['spread'].iloc[i-1]

			if method=='log_spread':
				bband_l		   = self.df['bband_l'].iloc[i]
				bband_u		   = self.df['bband_u'].iloc[i]
				self.df.loc[self.df.index[i],'signal'] = 1 if spread_shifted < spread and spread > bband_u else -1 if spread_shifted > spread and spread < bband_l else 0
				# self.df.loc[self.df.index[i],'signal'] = 1 if spread_shifted < spread and spread > bband_u else -1 if abs(spread)<0.0005 else 0
			if method=='Regression':
				entryZscore    = 1.5
				zscore		   = self.df['zScore'].iloc[i]
				zscore_shifted = self.df['zScore'].iloc[i-1]
				self.df.loc[self.df.index[i],'signal'] = 1 if zscore_shifted < zscore and zscore > entryZscore else -1 if zscore_shifted > zscore and zscore < -entryZscore else 0

			signal = self.df.loc[self.df.index[i],'signal']
			self.df.loc[self.df.index[i],'signal_english'] = 'scenario_A' if signal==1 else 'scenario_B' if signal==-1 else None

			if self.df['signal_english'].iloc[i]=='scenario_A' and status != 'looking for scenario_B':
				# When spread > bband_u, go scenario_A: Sell pair0, buy pair1.		# IN BLUE

				trades['nb_trades'] += 1

				# To simulate the spread, we buy/sell on the following minute candle (it has been shifted already, so it's on the same index).
				price_pair0_next_minute = Decimal(self.df[pairs[0]+'_m'].iloc[i+1])
				price_pair1_next_minute = Decimal(self.df[pairs[1]+'_m'].iloc[i+1])

				self.df.loc[self.df.index[i], 'sellprice_'+pairs[0]] = price_pair0_next_minute
				self.df.loc[self.df.index[i], 'buyprice_'+ pairs[1]] = price_pair1_next_minute
				self.df.loc[self.df.index[i], 'spread_trigger']      = self.df.loc[self.df.index[i],'spread']

				quantity_pair0_to_sell  = balance_pair0*alloc_pct/100
				quote_quantity_sell     = quantity_pair0_to_sell*price_pair0_next_minute
				fee_in_quote_sell       = quote_quantity_sell*Decimal(0.075)/Decimal(100)				# 2 operations, for the same quote_quantity, we pay 2 times the fees
				received_quote_quantity = quote_quantity_sell - fee_in_quote_sell						# What we get in quote from the sell

				if method=='log_spread':
					quantity_pair1_to_buy   = received_quote_quantity/price_pair1_next_minute						# quantity_pair1_to_buy = quantity_pair0_to_sell*ratio
					fee_in_pair1_buy        = quantity_pair1_to_buy*Decimal(0.075)/Decimal(100)
					received_quantity_pair1 = self.exchange.RoundToValidQuantity(pair=pairs[1], quantity=quantity_pair1_to_buy-fee_in_pair1_buy)
				elif method=='Regression':
					quantity_pair1_to_buy   = received_quote_quantity/Decimal(beta)/price_pair1_next_minute
					fee_in_pair1_buy        = quantity_pair1_to_buy*Decimal(0.075)/Decimal(100)
					received_quantity_pair1 = self.exchange.RoundToValidQuantity(pair=pairs[1], quantity=quantity_pair1_to_buy-fee_in_pair1_buy)

				balance_pair0 = Decimal('0')
				balance_pair1 += received_quantity_pair1
				balance_pair1_previous = self.df.loc[:, 'balance_pair1'].dropna().iloc[-1]

				self.df.loc[self.df.index[i], 'balance_pair0'] = balance_pair0
				self.df.loc[self.df.index[i], 'balance_pair1'] = balance_pair1

				if balance_pair1 < balance_pair1_previous:
					trades['nb_los_trades'] += 1
				else:
					trades['nb_win_trades'] += 1

				self.df.loc[self.df.index[i], 'fees'] = fee_in_quote_sell + fee_in_pair1_buy*price_pair1_next_minute
				self.df.loc[self.df.index[i], 'quotevalue_pair1'] = balance_pair1*price_pair1_next_minute

				status = 'looking for scenario_B'

			elif self.df['signal_english'].iloc[i]=='scenario_B' and status != 'looking for scenario_A':
				# When spread > bband_u, go scenario_B:  Sell pair1, buy pair0.		# IN ORANGE

				trades['nb_trades'] += 1

				# To simulate the spread, we buy/sell on the following minute candle
				price_pair0_next_minute = Decimal(self.df[pairs[0]+'_m'].iloc[i+1])
				price_pair1_next_minute = Decimal(self.df[pairs[1]+'_m'].iloc[i+1])


				self.df.loc[self.df.index[i], 'buyprice_'+ pairs[0]] = price_pair0_next_minute
				self.df.loc[self.df.index[i], 'sellprice_'+pairs[1]] = price_pair1_next_minute
				self.df.loc[self.df.index[i], 'spread_trigger']      = self.df.loc[self.df.index[i],'spread']


				quantity_pair1_to_sell  = balance_pair1*alloc_pct/100
				quote_quantity_sell     = quantity_pair1_to_sell*price_pair1_next_minute
				fee_in_quote_sell       = quote_quantity_sell*Decimal(0.075)/Decimal(100)
				received_quote_quantity = quote_quantity_sell - fee_in_quote_sell						# What we get in quote from the sell

				if method=='log_spread':
					quantity_pair0_to_buy   = received_quote_quantity/price_pair0_next_minute
					fee_in_pair0_buy        = quantity_pair0_to_buy*Decimal(0.075)/Decimal(100)
					received_quantity_pair0 = self.exchange.RoundToValidQuantity(pair=pairs[0], quantity=quantity_pair0_to_buy-fee_in_pair0_buy)
				elif method=='Regression':
					quantity_pair0_to_buy   = received_quote_quantity*Decimal(beta)/price_pair0_next_minute
					fee_in_pair0_buy        = quantity_pair0_to_buy*Decimal(0.075)/Decimal(100)
					received_quantity_pair0 = self.exchange.RoundToValidQuantity(pair=pairs[0], quantity=quantity_pair0_to_buy-fee_in_pair0_buy)

				balance_pair0 += received_quantity_pair0
				balance_pair0_previous = self.df.loc[:, 'balance_pair0'].dropna().iloc[-1]
				balance_pair1 = Decimal('0')

				self.df.loc[self.df.index[i], 'balance_pair0'] = balance_pair0
				self.df.loc[self.df.index[i], 'balance_pair1'] = balance_pair1

				if balance_pair0 < balance_pair0_previous:
					trades['nb_los_trades'] += 1
				else:
					trades['nb_win_trades'] += 1

				self.df.loc[self.df.index[i], 'fees'] = fee_in_quote_sell + fee_in_pair0_buy*price_pair0_next_minute
				self.df.loc[self.df.index[i], 'quotevalue_pair0'] = balance_pair0*price_pair0_next_minute

				status = 'looking for scenario_A'

			else:
				# Do nothing
				self.df.loc[self.df.index[i], 'fees'] = Decimal(0)


		# Compute profits and compare to buy and hold
		quotevalue_pair0_initiale = self.df.loc[:, 'quotevalue_pair0'].dropna().iloc[0]
		quotevalue_pair1_initiale = self.df.loc[:, 'quotevalue_pair1'].dropna().iloc[0]
		quotevalue_pair0_finale   = self.df.loc[:, 'quotevalue_pair0'].dropna().iloc[-1]
		quotevalue_pair1_finale   = self.df.loc[:, 'quotevalue_pair1'].dropna().iloc[-1]
		price_pair0_at_first_quotevalue = self.df.loc[self.df['quotevalue_pair0'] == quotevalue_pair0_initiale, pairs[0]+'_h'].iloc[0]
		price_pair0_at_last_quotevalue  = self.df.loc[self.df['quotevalue_pair0'] == quotevalue_pair0_finale,   pairs[0]+'_h'].iloc[0]
		price_pair1_at_first_quotevalue = self.df.loc[self.df['quotevalue_pair1'] == quotevalue_pair1_initiale, pairs[1]+'_h'].iloc[0]
		price_pair1_at_last_quotevalue  = self.df.loc[self.df['quotevalue_pair1'] == quotevalue_pair1_finale,   pairs[1]+'_h'].iloc[0]
		quote_profits_pair0 = (Decimal(quotevalue_pair0_finale) / quotevalue_pair0_initiale - 1)*100
		quote_profits_pair1 = (Decimal(quotevalue_pair1_finale) / quotevalue_pair1_initiale - 1)*100
		buy_hold_pair0 = (price_pair0_at_last_quotevalue / price_pair0_at_first_quotevalue - 1)*100
		buy_hold_pair1 = (price_pair1_at_last_quotevalue / price_pair1_at_first_quotevalue - 1)*100
		# print(f'quotevalue {pairs[0].replace(quote, "")} initiale = {round(quotevalue_pair0_initiale,8)}')
		# print(f'quotevalue {pairs[0].replace("quote, "")} finale   = {round(quotevalue_pair0_finale,8)}')
		print(f'{quote} profits of {pairs[0].replace(quote, "")}      = {round(quote_profits_pair0, 1)}%      (buy & hold = {round(buy_hold_pair0, 1)}%)')
		# print(f'quotevalue {pairs[1].replace(quote, "")} initiale = {round(quotevalue_pair1_initiale,8)}')
		# print(f'quotevalue {pairs[1].replace(quote, "")} finale   = {round(quotevalue_pair1_finale,8)}')
		print(f'{quote} profits of {pairs[1].replace(quote, "")}      = {round(quote_profits_pair1, 1)}%   (buy & hold = {round(buy_hold_pair1, 1)}%)')

		# Remove 0s & NaNs and compute metric
		temp    = self.df.loc[:,'quotevalue_pair0'].astype(float)
		cleaned = temp[np.where(temp, True, False)].dropna().pct_change()
		metric  = sharpe_ratio(cleaned, annualization=1)
		metric  = metric if abs(metric) != np.inf and not np.isnan(metric) else 0
		print('Sharpe Ratio : ', round(metric, 2))

		if plot:
			self.plotBotBacktest(quote	   = quote,
								 pairs	   = pairs,
								 length	   = length,
								 std	   = std,
								 alloc_pct = alloc_pct,
								 trades    = trades,
								 )

		return quote_profits_pair0, metric


	def plotBotBacktest(self, quote:str, pairs:list, length:int, std:float, alloc_pct:int, trades:dict):

		# Pairs and spread ______________________________________________________________________________________________________________________
		# max_indice = None
		max_indice = 500
		fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14,12))

		# First plot
		ax3 = ax1.twinx()
		ax1.plot(self.df.index[:max_indice], self.df[pairs[0]+'_h'].iloc[:max_indice], color='blue',  label=f"{pairs[0].replace('BTC', '')} price in {quote}")
		ax3.plot(self.df.index[:max_indice], self.df[pairs[1]+'_h'].iloc[:max_indice], color='green')
		ax1.plot([], [], color='green',   label=f"{pairs[1].replace(quote, '')} price in {quote}")	# Empty plot just for the label
		# Add the buys and sells
		ax1.scatter(self.df.index[:max_indice], self.df['buyprice_'+pairs[0]].iloc[:max_indice],  color='orange', marker='x', s=45)
		ax1.scatter(self.df.index[:max_indice], self.df['sellprice_'+pairs[0]].iloc[:max_indice], color='black',  marker='x', s=45)
		ax3.scatter(self.df.index[:max_indice], self.df['buyprice_'+pairs[1]].iloc[:max_indice],  color='orange', marker='x', s=45)
		ax3.scatter(self.df.index[:max_indice], self.df['sellprice_'+pairs[1]].iloc[:max_indice], color='black',  marker='x', s=45)
		ax1.plot([], [], color='orange', label="buys")			# Empty plot just for the label
		ax1.plot([], [], color='black',  label="sells")			# Empty plot just for the label
		# Legend and tites
		ax1.set_title(f'Pairs Trading   -   {self.timeframe}    -    Bbands length={length}, std={std} \n\nPrices of {pairs[0].replace(quote, "")} and {pairs[1].replace(quote, "")} in {quote}')
		ax1.legend(loc="upper left")
		ax1.set_ylabel(f'Price of {pairs[0].replace(quote, "")} in {quote}')
		ax3.set_ylabel(f'Price of {pairs[1].replace(quote, "")} in {quote}')
		ax1.tick_params(axis='y',  colors='blue')
		ax3.tick_params(axis='y',  colors='green')

		# Second plot
		ax4 = ax2.twinx()
		ax2.plot(self.df.index[:max_indice], self.df['spread'].iloc[:max_indice],  color='blue', label="Spread")
		# ax4.plot(self.df.index[:max_indice], self.df['zScore'].iloc[:max_indice], color='red')
		ax2.plot(self.df.index[:max_indice], self.df['bband_l'].iloc[:max_indice], color='red',  label="bbands")
		ax2.plot(self.df.index[:max_indice], self.df['bband_u'].iloc[:max_indice], color='red')
		ax2.scatter(self.df.index[:max_indice], self.df['spread_trigger'].iloc[:max_indice], c='black', marker='x', label='Actions')
		ax2.set_title(f'Spread - Bbands length={length}, std={std}')
		ax2.set_xlabel('Date')
		ax2.set_ylabel('Spread value')
		ax2.legend(loc="upper left")
		ax2.grid(linestyle='--', axis='y')
		plt.subplots_adjust(hspace = 10)
		plt.show()


		# Strategy evolution ______________________________________________________________________________________________________________________
		fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14,12))
		# First plot
		ax3 = ax1.twinx()
		ax1.scatter(self.df.index, self.df.loc[:, 'balance_pair0'].replace({0:np.nan}), color='blue',  label=f"{pairs[0].replace(quote, '')} balance", s=15)
		ax1.scatter([], [], 	 					         							color='green', label=f"{pairs[1].replace(quote, '')} balance", s=15)	# Empty plot just for the label
		ax3.scatter(self.df.index, self.df.loc[:, 'balance_pair1'].replace({0:np.nan}), color='green', s=15)
		ax1.set_title(f'Pairs Trading   -    {self.timeframe}    -    Bbands length={length}, std={std}    -    alloc={alloc_pct}%    -    Trades : {trades["nb_trades"]}\n\nQuantity of coins held')
		ax1.legend(loc="upper left")
		ax1.set_ylabel(f'Quantity of {pairs[0].replace(quote, "")} held')
		ax3.set_ylabel(f'Quantity of {pairs[1].replace(quote, "")} held')
		ax1.tick_params(axis='y',  colors='blue')
		ax3.tick_params(axis='y',  colors='green')
		ax1.axhline(self.df['balance_pair0'].dropna().iloc[0], color='blue',  linestyle='--')
		ax3.axhline(self.df['balance_pair1'].dropna().iloc[0], color='green',  linestyle='--')
		# Second plot
		ax2.scatter(self.df.index, self.df.loc[:, 'quotevalue_pair0'],  color='blue',  label=f"{pairs[0].replace(quote, '')} holdings value in {quote}", marker='x', s=15)
		ax2.scatter(self.df.index, self.df.loc[:, 'quotevalue_pair1'],  color='green', label=f"{pairs[1].replace(quote, '')} holdings value in {quote}", marker='x', s=15)
		ax2.plot(self.df.index, self.df['fees'].astype(float).cumsum(), color='black', label="Cumulative fees (already included)", linestyle=':')
		ax2.set_title(f'Value of holdings in {quote}')
		ax2.legend(loc="upper left")
		ax2.set_xlabel('Date')
		ax2.set_ylabel(f'Value of holdings in {quote}')
		ax2.grid(linestyle='--', axis='y')
		ax2.axhline(self.df['quotevalue_pair0'].dropna().iloc[0], color='blue',  linestyle='--')
		ax2.axhline(self.df['quotevalue_pair1'].dropna().iloc[0], color='green',  linestyle='--')
		plt.subplots_adjust(hspace=10)
		plt.show()

		# Ajouter zero line pour les deux y axes du 1er plot, dans la couleur correspondante

		return



if __name__ == '__main__':

	backtester = BackTestingPairsTrading('1D')

	backtester.backtest(quote     = 'BTC',
						pairs     = ['ETCBTC', 'XRPBTC'],
						starting_balances = dict(quote=1, quote_for_pair0=0.01, quote_for_pair1=0.01),
						length    = 70,
						std       = 2,
						alloc_pct = 100,		# How much of the coin balance we bet each time
						plot      = True
						)


	# results           = dict()
	# results['length'] = []
	# results['std']    = []
	# results['metric'] = []
	# results['quote_profits_pair0'] = []
	#
	# # Find the best length and std for this timeframe
	# for length_ in np.linspace(start=5, stop=90, num=5):
	# 	for std_ in np.linspace(start=1, stop=3, num=5):
	#
	# 		quote_profits_pair0, metric_ = backtester.backtest(quote     = 'BTC',
	# 														   pairs     = ['ETHBTC', 'XRPBTC'],
	# 														   starting_balances = dict(quote=1,pair0=1,pair1=1),
	# 														   length    = length_,
	# 														   std       = std_,
	# 														   alloc_pct = 100,
	# 														 )
	#
	# 		print(f'length, std = {length_}, {std_}')
	# 		print(f'metric = ', metric_)
	# 		print(f'quote profits on ETH = {round(quote_profits_pair0, 2)}%')
	# 		print('________________________')
	#
	# 		results['length'].append(length_)
	# 		results['std'].append(std_)
	# 		results['metric'].append(metric_)
	# 		results['quote_profits_pair0'].append(quote_profits_pair0)
	#
	#
	# # ______________________________________________
	# # Plot the results of the grid search
	# df_ = pd.DataFrame({'length': results['length'],
	# 				   'std'    : results['std'],
	# 				   'metric' : results['metric'],
	# 				   'quote_profits_pair0': results['quote_profits_pair0']})
	# # re-create the 2D-arrays
	# x1 = np.linspace(df_['length'].min(), df_['length'].max(), len(df_['length'].unique()))
	# y1 = np.linspace(df_['std'].min(),    df_['std'].max(),    len(df_['std'].unique()))
	# x2, y2 = np.meshgrid(x1, y1)
	# z2 = griddata((df_['length'], df_['std']), df_['metric'], (x2, y2), method='cubic')
	# fig  = plt.figure(figsize = (16,12))
	# ax   = fig.gca(projection='3d')
	# surf = ax.plot_surface(x2, y2, z2)
	# ax.set_xlabel('length')
	# ax.set_ylabel('std')
	# ax.set_zlabel('metric')
	# plt.title('[ETHBTC, XRPBTC] 2h sharpe_ratio estimations')
	# plt.show()
	#
	# import plotly.graph_objs as go
	# fig = go.Figure()
	# fig.add_trace(go.Scatter3d(x=df_.loc[:,'length'], y=df_.loc[:,'std'], z=df_.loc[:,'metric'],
	# 						   mode='markers',
	# 						   marker=dict(
	# 									size       = 5,
	# 									color      = df_.loc[:,'quote_profits_pair0'],      # set color to an array/list of desired values
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
	# 				)
	# fig.show()