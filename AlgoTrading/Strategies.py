from Exchange     import Binance
import pandas as pd
import pandas_ta as ta
import numpy as np
import statistics
from yaspin import yaspin
from scipy.optimize import minimize

exchange = Binance(filename = 'credentials.txt')



def maCrossoverStrategy(df, i:int, signal:str, fast_period:int, slow_period:int)->bool:
	""" If price is 10% below the Slow MA, return True """

	fast_ema = "EMA_" + str(fast_period)
	slow_ema = "EMA_" + str(slow_period)

	try:
		# Add the necessary indocators if it's not already here
		if not df.__contains__(fast_ema) and not df.__contains__(slow_ema):
			df[fast_ema] = ta.ema(df['close'], length=fast_period)
			df[slow_ema] = ta.ema(df['close'], length=slow_period)

		if signal == 'buy':
			if i > 0 and df[fast_ema][i-1] >= df[slow_ema][i-1] and df[fast_ema][i] < df[slow_ema][i]:
				return True

		if signal == 'sell':
			if i > 0 and df[fast_ema][i-1] <= df[slow_ema][i-1] and df[fast_ema][i] > df[slow_ema][i]:
				return True

	except Exception as e:
		# print("Skipping the pair.")		# On a parfois pas assez de data pour calculer les indicateurs : data_len < period.
		return False


strategies_dict = {'maCrossoverStrategy': maCrossoverStrategy}




# if __name__ == "__main__":
# 	quote           = 'BTC'
# 	pair            = 'ADABTC'
# 	timeframe       = '1h'
