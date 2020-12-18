import Strategies
from Exchange     import Binance

from decimal  import Decimal, getcontext
import plotly.graph_objs as go
import pandas as pd
from plotly.subplots import make_subplots
import numpy as np
import matplotlib.pyplot as plt



class BackTesting:
	""" Class used in 'BotDeMoi' to backtest a strategy. """

	def __init__(self, exchange, strategy):
		self.exchange  = exchange		# To get the candle data
		self.strategy  = strategy		# To compute the indicators
		self.timeframe = '2h'

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



	def Backtest(self, quote:str, pair:str, starting_quote_balance:float, alloc_pct:int):
		""" Also used in Dashboard_BT """

		# Get the dataframe from the csv file
		print("_________ " + pair + " _________")
		df = pd.read_csv(f'historical_data/{quote}/{pair}_{self.timeframe}', sep='\t', na_values='-')

		df.loc[df.buys.isna(),  'buys']  = 0
		df.loc[df.buys != 0,    'buys']  = 1
		df.loc[df.sells.isna(), 'sells'] = 0
		df.loc[df.sells != 0,   'sells'] = 1

		pair_results  = dict()
		last_buy 	  = None
		quote_balance = Decimal(starting_quote_balance)
		df['backtest_buys']    = np.nan
		df['backtest_sells']   = np.nan
		df['backtest_fees']    = np.nan
		df['backtest_balance'] = np.nan


		# Go through all candlesticks
		for i in range(0, len(df['close'])-1):

			# If we didn't already buy :
			if last_buy is None:
				# Check for a buy signal
				if df['buys'].loc[i]==1.0:
					time     	     = df['time'][i]		# str
					price            = df['close'][i]		# numpy.float64
					quote_alloc      = quote_balance*alloc_pct/100					# Buy for alloc_pct of the quote balance
					fee_in_quote     = quote_alloc*Decimal(0.075)/Decimal(100)
					rea_quote_alloc  = quote_alloc - fee_in_quote					# Substract the fees to get the quote value of the transaction
					quantity_to_buy  = quote_alloc/Decimal(price)
					quote_balance    -= rea_quote_alloc								# Should be zero but rounding errors
					# quote_balance = 0
					df.loc[df.index[i], 'quote_balance'] = quote_balance
					df.loc[df.index[i], 'backtest_buys'] = price
					df.loc[df.index[i], 'backtest_fees'] = fee_in_quote

					last_buy  = {"index":i, "price":price, "quantity":quantity_to_buy}
					print(f'{time} - Buy   {round(quantity_to_buy, 2)} {pair[:3]} at {price} {quote}')
					# print(quote_balance)

			# If we already bought :
			elif last_buy is not None and i > last_buy["index"]+1:
				# Check for a sell signal
				if df['sells'].loc[i]==1.0:
					time     	     = df['time'][i]
					price            = df['close'][i]
					quantity_to_sell = last_buy.get('quantity')						# Sell everything
					quote_quantity   = quantity_to_sell*Decimal(price)
					fee_in_quote     = quote_quantity*Decimal(0.075)/Decimal(100)
					quote_balance    += quote_quantity - fee_in_quote
					df.loc[df.index[i], 'quote_balance'] = quote_balance
					df.loc[df.index[i], 'backtest_buys'] = price
					df.loc[df.index[i], 'backtest_fees'] = fee_in_quote

					last_buy  = None
					print(f'{time} - Sell  {round(quantity_to_sell, 2)} {pair[:3]} at {price} {quote}. Balance = {round(quote_balance,3)} {quote}')


		return df


	@staticmethod
	def plotBotBacktest(df_, buys, sells):

		# # Create figure with secondary y-axis for the volume
		# fig = make_subplots(shared_xaxes=True, specs=[[{"secondary_y": True}]])
		#
		# # plot the close prices for this pair
		# fig.add_trace(go.Scatter(x    = df['time'],
		# 						 y    = df['close'],
		# 						 mode = 'lines',
		# 						 name = 'Close prices',
		# 						 line = dict(color='rgb(255,255,51)', width=1.5)),
		# 			  secondary_y = False)
		#
		# # Add the volume
		# fig.add_trace(go.Bar(x       = df['time'],
		# 					 y       = df['volume'],
		# 					 name    = "Volume",
		# 					 marker  = dict(color='#a3a7b0')),
		# 			  secondary_y = True)
		#
		# # # Plot indicators on this pair
		# # for colname in df.columns[6:]:
		# # 	fig.add_trace(go.Scatter(x    = df['time'],
		# # 						     y    = df[colname],
		# # 						     name = colname))
		#
		# # Plot buy points on this pair
		# fig.add_trace(go.Scatter(
		# 						 x      = df['time'],
		# 						 y      = df['backtest_buys'],
		# 						 # x 	    = [item[0] for item in buys],
		# 						 # y 	    = [item[1] for item in buys],
		# 						 name   = 'Buy Signals',
		# 						 mode   = 'markers',
		# 						 marker = dict(size   = 5,
		# 									   color  = 'red',
		# 									   symbol = 'cross',
		# 									   line   = dict(width=1.5, color='red'))))
		#
		# # Plot sell points on this pair
		# fig.add_trace(go.Scatter(
		# 						 x      = df['time'],
		# 						 y      = df['backtest_sells'],
		# 						 # x 	   = [item[0] for item in sells],
		# 						 # y 	   = [item[1] for item in sells],
		# 						 name   = 'Sell Signals',
		# 						 mode   = 'markers',
		# 						 marker = dict(size  = 5,
		# 									   color = 'green',
		# 									   line  = dict(width=1.5, color='green'))))
		#
		# # Layout for the  graph
		# fig.update_layout({
		# 		"margin": {"t": 30, "b": 20},
		# 		"height": 800,
		# 		"hovermode": "x",
		#
		# 		"xaxis"  : {
		# 			# "fixedrange"    : True,                     # Determines whether or not this axis is zoom-able. Iftrue, then zoom is disabled.
		# 			"range"         : [df['time'].to_list()[0], df['time'].to_list()[-1]],
		# 			"showline"      : True,
		# 			"zeroline"      : False,
		# 			"showgrid"      : False,
		# 			"showticklabels": True,
		# 			"rangeslider"   : {"visible": False},
		# 			"showspikes"    : True,
		# 			"spikemode"     : "across+toaxis",
		# 			"spikesnap"     : "cursor",
		# 			"spikethickness": 0.5,
		# 			"color"         : "#a3a7b0",
		# 		},
		# 		"yaxis"  : {
		# 			# "autorange"      : True,
		# 			# "rangemode"     : "normal",
		# 			# "fixedrange"    : False,
		# 			"showline"      : False,
		# 			"showgrid"      : False,
		# 			"showticklabels": True,
		# 			"ticks"         : "",
		# 			"showspikes"    : True,
		# 			"spikemode"     : "across+toaxis",
		# 			"spikesnap"     : "cursor",
		# 			"spikethickness": 0.5,
		# 			"spikecolor"    : '#a3a7b8',
		# 			"color"         : "#a3a7b0",
		# 		},
		# 		"yaxis2" : {
		# 			# "fixedrange"    : True,
		# 			"showline"      : False,
		# 			"zeroline"      : False,
		# 			"showgrid"      : False,
		# 			"showticklabels": True,
		# 			"ticks"         : "",
		# 			# "color"        : "#a3a7b0",
		# 			"range"         : [0, df['volume'].max() * 10],
		# 		},
		# 		"legend" : {
		# 			"font"          : dict(size=15, color="#a3a7b0"),
		# 		},
		# 		"plot_bgcolor"  : "#23272c",
		# 		"paper_bgcolor" : "#23272c",
		#
		# 		})
		#
		#
		# fig.show()

		plt.figure(figsize = (16,12))
		plt.scatter(df_.index, df_.quote_balance, color='blue')
		plt.xlabel('Date')
		plt.ylabel('BTC balance')
		plt.title('Optimal buys and sells results')
		plt.show()



if __name__ == '__main__':

	exchange   = Binance(filename='credentials.txt')
	backtester = BackTesting(exchange, strategy='test')


	df = backtester.Backtest(quote='BTC',pair='ETHBTC', starting_quote_balance=1, alloc_pct=40)

	backtester.plotBotBacktest(df_   = df,
							   buys  = df.loc[:,'backtest_buys'],
							   sells = df.loc[:,'backtest_sells'])