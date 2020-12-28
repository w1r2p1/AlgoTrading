from Strategies import *

from decimal  import Decimal, getcontext
import plotly.graph_objs as go
import pandas as pd
from plotly.subplots import make_subplots
import numpy as np



class BackTesting:
	""" Class used in 'BotDeMoi' to backtest a strategy. """

	def __init__(self, exchange, strategy):
		self.exchange  = exchange		# To get the candle data
		self.strategy  = strategy		# To compute the indicators
		self.timeframe = '1h'

		self.complete_starting_balance  = Decimal(1)								# The quantity of quoteAsset we have. It will be spread equally between the pairs.
		self.complete_resulting_balance = Decimal(self.complete_starting_balance)
		self.profitable_symbols			= 0											# Number of profitable symbols
		self.unprofitable_symbols		= 0


	def BacktestAllPairs(self, pairs:dict, plot_data:bool):	 # object of the class 'StrategyEvaluator'

		options = dict(starting_balance		 = self.complete_starting_balance/len(pairs),
					   initial_profits 	 	 = 1.012,
					   initial_stop_loss 	 = 0.9,
					   incremental_profits   = 1.006,
					   incremental_stop_loss = 0.996)

		print("Complete starting balance : " + str(round(self.complete_starting_balance, 3)))

		for quote in pairs.keys():
			print("QuoteAsset: ", quote)

			for pair in pairs[quote]:
				print("_____________\n", pair)
				print("Starting balance : " + str(round(options['starting_balance'], 3)) + " " + pair[-3:])
				pair_results = self.Backtest(quote			  = quote,
											 pair			  = pair,
											 # plot			  = plot_data,
											 starting_balance = options['starting_balance'])

				if pair_results['status'] == 'looking to enter':	# if the last order of a bot is a buy, its balance is 0 and we don't need to compute this.
					if pair_results['balance'] > options['starting_balance']:
						self.profitable_symbols   = self.profitable_symbols + 1
					else:
						self.unprofitable_symbols = self.unprofitable_symbols + 1

					print("\nStarting balance: "  + str(round(options['starting_balance'], 3)) + " " + pair[-3:] + ", resulting balance: " + str(round(pair_results['balance'], 3)) + " " + pair[-3:])
					print("difference : " + str(round(pair_results['balance']-options['starting_balance'], 3)) + " " + pair[-3:])
					self.complete_resulting_balance += (pair_results['balance']-options['starting_balance'])
					print("Complete_resulting_balance = " + str(round(self.complete_resulting_balance, 3)) + " " + pair[-3:])

				elif pair_results['status'] == 'looking to exit':
					print("\nThe bot last bought, complete balance unchanged = ", round(self.complete_resulting_balance, 3))


			print("____________________________________________________________________")
			print(self.strategy + " stats :")
			print("\nProfitable Symbols : " + str(self.profitable_symbols))
			print("Unprofitable Symbols : " + str(self.unprofitable_symbols))

			if self.profitable_symbols > 0:
				profitability = Decimal(100.0) * (self.complete_resulting_balance/self.complete_starting_balance - Decimal(1.0))
				print("\nWith an initial balance of "	+ str(self.complete_starting_balance) +
					  " and a final balance of "	+ str(round(self.complete_resulting_balance, 3)))
				print("the profitability over "	+ str(len(pairs)) + " pairs is "	+ str(round(profitability, 2))+"%.")



	def Backtest(self, quote:str, pair:str, starting_balance:float):
		""" Also used in Dashboard_BT """

		# Get the dataframe from the csv file
		print("_________ " + pair + " _________")
		filename = 'Historical_data/' + quote + '/' + pair + '_' + self.timeframe
		df = pd.read_csv(filename, sep='\t')

		df_bt = df[['time','close']]

		pair_results = dict()
		last_buy 	 = None
		balance  	 = Decimal(starting_balance)
		# df['backtest_buys']    = Decimal(np.nan)
		# df['backtest_sells']   = Decimal(np.nan)
		# df['backtest_fees']    = Decimal(np.nan)
		# df['backtest_balance'] = Decimal(np.nan)
		df_bt['backtest_buys']    = np.nan
		df_bt['backtest_sells']   = np.nan
		df_bt['backtest_fees']    = np.nan
		df_bt['backtest_balance'] = np.nan
		# df['backtest_buys']    = Decimal('0')
		# df['backtest_sells']   = Decimal('0')
		# df['backtest_fees']    = Decimal('0')
		# df['backtest_balance'] = Decimal('0')



		# Use this to backtest with np.where
		# df['backtest_buys'] = np.where((df['EMA_10'].shift(-1)<=df['EMA_50'].shift(-1)) & (df['EMA_10']>=df['EMA_50']), df['close'], np.nan)
		# print(df['backtest_buys'].dropna())

		# Go through all candlesticks
		for i in range(0, len(df_bt['close'])-1):

			# If we didn't already buy :
			if last_buy is None:
				# Check for a buy signal
				buy_signal = strategies_dict[self.strategy](df=df_bt, i=i, signal='buy', fast_period=10, slow_period=50)

				if buy_signal:
					time     	   = df_bt['time'][i]		# str
					price          = df_bt['close'][i]		# numpy.float64
					quantity       = balance/Decimal(price)
					quote_quantity = Decimal(price)*quantity
					fee_in_quote   = quote_quantity*Decimal(0.075)/Decimal(100)
					df_bt['backtest_buys'].iloc[i] = price
					df_bt['backtest_fees'].iloc[i] = fee_in_quote

					last_buy  = {"index":i, "price":price, "quantity":quantity}
					print(time + " - Buy  " + str(round(quantity, 2)) + " " + pair[:3] + " at " + str(price) + " " + quote)
					balance   = 0
					# pair_results['status'] = 'looking to exit'
				# else:
				# 	df['backtest_balance'].iloc[i] = np.nan

			# If we already bought :
			elif last_buy is not None and i > last_buy["index"]+1:
				# Check for a sell signal
				sell_signal = strategies_dict[self.strategy](df=df_bt, i=i, signal='sell', fast_period=10, slow_period=50)

				if sell_signal:
					time     	   = df_bt['time'][i]
					price          = df_bt['close'][i]
					# print(price, type(price), float(price), type(float(price)))
					quantity       = last_buy.get('quantity')
					quote_quantity = Decimal(price)*quantity
					fee_in_quote   = quote_quantity*Decimal(0.075)/Decimal(100)
					df_bt['backtest_sells'].iloc[i] = price
					df_bt['backtest_fees'].iloc[i]  = fee_in_quote

					balance = quote_quantity
					df_bt['backtest_balance'].iloc[i] = balance
					print(time + " - Sell " + str(round(quantity, 2)) + " " + pair[:3] + " at " + str(price) + ". Balance : " + str(round(balance,3)) + " " + quote)
					last_buy  = None
					# pair_results['status'] = 'looking to enter'

		# pair_results['backtest_buys']    = df['backtest_buys']			# Decimal
		# pair_results['backtest_sells']   = df['backtest_sells']
		# pair_results['backtest_fees']    = df['backtest_fees']
		# pair_results['backtest_balance'] = df['backtest_balance']		# Decimal


		return df_bt


	@staticmethod
	def plotBotBacktest(df, buys:list, sells:list):

		# Create figure with secondary y-axis for the volume
		fig = make_subplots(shared_xaxes=True, specs=[[{"secondary_y": True}]])

		# plot the close prices for this pair
		fig.add_trace(go.Scatter(x    = df['time'],
								 y    = df['close'],
								 mode = 'lines',
								 name = 'Close prices',
								 line = dict(color='rgb(255,255,51)', width=1.5)),
					  secondary_y = False)

		# Add the volume
		fig.add_trace(go.Bar(x       = df['time'],
							 y       = df['volume'],
							 name    = "Volume",
							 marker  = dict(color='#a3a7b0')),
					  secondary_y = True)

		# # Plot indicators on this pair
		# for colname in df.columns[6:]:
		# 	fig.add_trace(go.Scatter(x    = df['time'],
		# 						     y    = df[colname],
		# 						     name = colname))

		# Plot buy points on this pair
		if buys:
			fig.add_trace(go.Scatter(x 	   = [item[0] for item in buys],
									 y 	   = [item[1] for item in buys],
									 name   = 'Buy Signals',
									 mode   = 'markers',
									 marker = dict(size   = 10,
												   color  = 'red',
												   symbol = 'cross',
											  	   line   = dict(width=1.5, color='red'))))

		# Plot sell points on this pair
		if sells:
			fig.add_trace(go.Scatter(x 	   = [item[0] for item in sells],
									 y 	   = [item[1] for item in sells],
									 name   = 'Sell Signals',
									 mode   = 'markers',
									 marker = dict(size  = 10,
												   color = 'green',
												   line  = dict(width=1.5, color='green'))))

		# Layout for the  graph
		fig.update_layout({
				"margin": {"t": 30, "b": 20},
				"height": 800,
				"hovermode": "x",

				"xaxis"  : {
					# "fixedrange"    : True,                     # Determines whether or not this axis is zoom-able. Iftrue, then zoom is disabled.
					"range"         : [df['time'].to_list()[0], df['time'].to_list()[-1]],
					"showline"      : True,
					"zeroline"      : False,
					"showgrid"      : False,
					"showticklabels": True,
					"rangeslider"   : {"visible": False},
					"showspikes"    : True,
					"spikemode"     : "across+toaxis",
					"spikesnap"     : "cursor",
					"spikethickness": 0.5,
					"color"         : "#a3a7b0",
				},
				"yaxis"  : {
					# "autorange"      : True,
					# "rangemode"     : "normal",
					# "fixedrange"    : False,
					"showline"      : False,
					"showgrid"      : False,
					"showticklabels": True,
					"ticks"         : "",
					"showspikes"    : True,
					"spikemode"     : "across+toaxis",
					"spikesnap"     : "cursor",
					"spikethickness": 0.5,
					"spikecolor"    : '#a3a7b8',
					"color"         : "#a3a7b0",
				},
				"yaxis2" : {
					# "fixedrange"    : True,
					"showline"      : False,
					"zeroline"      : False,
					"showgrid"      : False,
					"showticklabels": True,
					"ticks"         : "",
					# "color"        : "#a3a7b0",
					"range"         : [0, df['volume'].max() * 10],
				},
				"legend" : {
					"font"          : dict(size=15, color="#a3a7b0"),
				},
				"plot_bgcolor"  : "#23272c",
				"paper_bgcolor" : "#23272c",

				})


		fig.show()