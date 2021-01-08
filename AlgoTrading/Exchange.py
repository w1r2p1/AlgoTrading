import requests
import json
import hmac
import time
from datetime import datetime, timedelta
import pandas as pd
import hashlib
from decimal import Decimal
import math
import win32api
import re
from tqdm import tqdm


class Binance:

	ORDER_STATUS_NEW 				= 'NEW'
	ORDER_STATUS_PARTIALLY_FILLED 	= 'PARTIALLY_FILLED'
	ORDER_STATUS_FILLED 			= 'FILLED'
	ORDER_STATUS_CANCELED 			= 'CANCELED'
	ORDER_STATUS_PENDING_CANCEL 	= 'PENDING_CANCEL'
	ORDER_STATUS_REJECTED 			= 'REJECTED'
	ORDER_STATUS_EXPIRED 			= 'EXPIRED'

	SIDE_BUY  = 'BUY'
	SIDE_SELL = 'SELL'

	ORDER_TYPE_LIMIT 			 = 'LIMIT'
	ORDER_TYPE_MARKET 			 = 'MARKET'
	ORDER_TYPE_STOP_LOSS 		 = 'STOP_LOSS'
	ORDER_TYPE_STOP_LOSS_LIMIT 	 = 'STOP_LOSS_LIMIT'
	ORDER_TYPE_TAKE_PROFIT 		 = 'TAKE_PROFIT'
	ORDER_TYPE_TAKE_PROFIT_LIMIT = 'TAKE_PROFIT_LIMIT'
	ORDER_TYPE_LIMIT_MAKER 		 = 'LIMIT_MAKER'

	KLINE_INTERVALS = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']

	def __init__(self, filename=None):

		self.base = 'https://api.binance.com'
		self.websocket = 'wss://stream.binance.com:9443'

		self.endpoints = {"order"			: '/api/v3/order',
						  "testOrder"		: '/api/v3/order/test',		# Test new order creation and signature/recvWindow long. Creates and validates a new order but does not send it into the matching engine.
						  "OCO"				: '/api/v3/order/OCO',		# One-Cancels-the-Other
						  "allOrders"		: '/api/v3/allOrders',
						  "klines"			: '/api/v3/klines',
						  "exchangeInfo"	: '/api/v3/exchangeInfo',
						  "24hrTicker" 		: '/api/v3/ticker/24hr',
						  "lastestPrice"    : '/api/v3/ticker/price',
						  "averagePrice"	: '/api/v3/avgPrice',
						  "orderBook" 		: '/api/v3/depth',
						  "trades"  		: '/api/v3/trades',
						  "account" 		: '/api/v3/account',
						  "time"			: '/api/v3/time',
						  }

		if filename is None:
			return
		f = open(filename, "r")
		contents = []
		if f.mode == 'r':
			contents = f.read().split('\n')

		self.binance_keys = dict(api_key = contents[0], secret_key=contents[1])
		self.headers 	  = {"X-MBX-APIKEY": self.binance_keys['api_key']}


	@staticmethod
	def _get(url, params=None, headers=None) -> dict:
		""" Makes a Get Request """
		try: 
			response = requests.get(url, params=params, headers=headers)
			data = json.loads(response.text)
		except Exception as e:
			print("Exception occurred when trying to access " + url)
			print(e)
			data = {'code': '-1', 'url':url, 'msg': e}
		return data


	@staticmethod
	def _post(url, params=None, headers=None) -> dict:
		""" Makes a Post Request """
		try: 
			response = requests.post(url, params=params, headers=headers)
			data = json.loads(response.text)
			data['url'] = url
		except Exception as e:
			print("Exception occurred when trying to access " + url)
			print(e)
			data = {'code': '-1', 'url':url, 'msg': e}
		return data


	def signRequest(self, params:dict):
		""" Signs the request to the Binance API """

		query_string = '&'.join(["{}={}".format(d, params[d]) for d in params])
		signature    = hmac.new(self.binance_keys['secret_key'].encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256)
		params['signature'] = signature.hexdigest()


	def GetServerTime(self):
		""" Test connectivity to the Rest API and get the current server time. """

		url  		= self.base + self.endpoints["time"]
		server_time = self._get(url)

		if server_time.__contains__('code'):
			return 0

		return server_time['serverTime']			# 1499827319559 timestamp in milliseconds


	def TimeDifferenceWithBinance(self)->int:
		""" Returns the clock difference between Binance and the computer in milliseconds """

		return  int(self.GetServerTime() - time.time()*1000)		# 350


	def SynchronizeToBinanceTime(self):

		binance_time = self.GetServerTime()
		print(binance_time)
		if binance_time is not None:
			# SetSystemTime takes time as argument in UTC time. UTC time is obtained using utcfromtimestamp()
			utcTime = datetime.utcfromtimestamp(binance_time/1000)
			win32api.SetSystemTime(utcTime.year, utcTime.month, utcTime.weekday(), utcTime.day, utcTime.hour, utcTime.minute, utcTime.second, 0)
			# Local time is obtained using fromtimestamp()
			localTime = datetime.now()
			print("Time updated to: " + localTime.strftime("%Y-%m-%d %H:%M") + " from Binance")

		else:
			print("Could not find time from Binance.")


	def GetNameOfPairs_WithQuoteasset(self, quotes:list=None) -> dict:
		""" Gets just the name the of pairs which have their quoteasset in 'quoteAssets',  tradable or not.
			-> # {'BTC':['ETHBTC', 'LTCBTC',...], 'ETH':[]} """

		url  = self.base + self.endpoints["exchangeInfo"]
		data = self._get(url)

		if data.__contains__('code'):
			return {}

		pairs_with_quoteassets = dict()

		for quote in quotes:
			pairs_list = []
			for pair in data['symbols']:
				if pair['quoteAsset'] in quote:
					pairs_list.append(pair['symbol'])
			pairs_with_quoteassets[quote] = pairs_list

		return pairs_with_quoteassets			# {'BTC':['ETHBTC', 'LTCBTC',...], 'ETH':[]}


	def GetMetadataOfPair(self, pair:str) -> dict:
		""" Gets the metadata of a pair
			-> {'symbol': 'ETHBTC', 'status': 'TRADING',...} """

		url  = self.base + self.endpoints["exchangeInfo"]
		data = self._get(url)

		if data.__contains__('code'):
			return {}

		for pair_info in data['symbols']:
			if pair_info['symbol'] == pair:
				return pair_info					# {'symbol': 'ETHBTC', 'status': 'TRADING', 'baseAsset': 'ETH', 'baseAssetPrecision': 8,.....}


	def GetMinNotional(self, pair)->str:
		""" Get the minimum notional value allowed for an order on a pair.
		 	minNotional = price*quantity (average price for market orders). """

		pair_info = self.GetMetadataOfPair(pair)


		if pair_info:
			for filtr in pair_info['filters']:
				if filtr['filterType'] == 'MIN_NOTIONAL':
					return filtr['minNotional']
		else:
			return '0'


	def GetAccountData(self) -> dict:
		""" Gets Balances & Account Data """

		url = self.base + self.endpoints["account"]
		
		params = {'recvWindow': 6000,					# Not mandatory
				  'timestamp' : self.GetServerTime()}	# Mandatory

		self.signRequest(params)
		account = self._get(url, params, self.headers)

		if account.__contains__('code'):
			return {}

		return account		# dict of all the data about the account : {'makerCommission': 10, 'takerCommission': 10, ...., 'balances': []}


	def GetAccountBalance(self, quote:str):

		account_data = self.GetAccountData()

		if account_data == {}:
			return f"Could not get account data on {quote} from Binance."

		balances = account_data['balances']
		for balance in balances:
			if balance['asset'] in quote:
				return balance


	def GetPairKlines(self, pair:str, timeframe:str, **kwargs):
		""" Gets the lastest candle data from Binance.
		 	The number of candles or the start date (in timestamp or datetime) can be specified"""

		candles = None
		parsed_timeframe = None

		# When the starting date is specified, compute the corresponding number of candles
		if 'start_date' in kwargs:
			start = kwargs['start_date']
			# Convert the timestamp to a datetime if needed
			if type(start)==int:
				start_date = datetime.utcfromtimestamp(start)
			else:
				start_date = start
			now = datetime.utcnow()

			parsed_timeframe = re.findall(r'[A-Za-z]+|\d+', timeframe)      # Separates '30m' in ['30', 'm']
			# Compute the number of candles to plot
			if parsed_timeframe[1].lower() == 'm':
				candles = (now - start_date).total_seconds() / 60.0    / float(parsed_timeframe[0])
			elif parsed_timeframe[1].lower() == 'h':
				candles = (now - start_date).total_seconds() / 3600.0  / float(parsed_timeframe[0])
			elif parsed_timeframe[1].lower() == 'd':
				candles = (now - start_date).total_seconds() / 86400.0 / float(parsed_timeframe[0])

		# When the number of candles is specified
		if 'candles' in kwargs:
			candles = kwargs['candles']
			candles = candles+1

		candles = math.ceil(candles)						# Round the number upward to its nearest integer
		# print(f"Retrieving {candles} candles.")

		if candles < 1000:
			initial_candles = candles
		else:
			initial_candles = 1000

		# Get the 1st set of candles
		params = f'?&symbol={pair}&interval={timeframe}&limit={initial_candles}'
		url    = self.base + self.endpoints['klines'] + params
		data    = self._get(url)

		if data.__contains__('code'):
			print("In 'Exchange' : Could not get the dataframe from Binance.")
			print(data)
			return pd.DataFrame()					# Return an empty dataframe if we couldn't get the data from Binance

		df = pd.DataFrame.from_dict(data)

		# if candles > 1000, repeat and append until we have everything.
		if candles > 1000:

			# # finds the biggest candles set <= 1000, in order to reduce the number of calls to Binance while getting the exact number of candles desired.
			# candles_to_generate = candles-1001
			# largest_divisor 	= gcd(1000, candles_to_generate)
			#
			# for candles_set in range(1001, candles, largest_divisor):
			# 	print("Getting an other set of " + str(largest_divisor) + " candles, starting at " + str(candles_set) + "...")
			# 	params 	= '?&symbol=' + pair + '&interval=' + timeframe + '&limit=' + str(largest_divisor) + '&endTime=' + str(df[df.columns[0]][0])
			# 	url 	= self.base + self.endpoints['klines'] + params
			# 	data 	= self._get(url)
			# 	df2 	= pd.DataFrame.from_dict(data)
			# 	df.append(df2, ignore_index = True)


			candles_to_generate = candles-1001
			# print(candles, candles_to_generate)
			jeu = min(1000, int(candles_to_generate/5))
			# jeu = min(1000, candles_to_generate)
			# for candles_set in range(1001, candles, jeu):
			for candles_set in tqdm(range(1001, candles, jeu), disable=kwargs.get('disable_tqdm', True)):
				# print(f"Need {candles} candles, have {len(df)}. Getting an other set of {jeu} candles, starting at {candles_set}...")
				# Stop one line earlier to avoid duplicated lines at edges. Remove one timeframe at the end.
				dt     = int(parsed_timeframe[0])
				m_h_d  = parsed_timeframe[1].lower()
				timedt = timedelta(minutes=dt) if m_h_d=='m' else timedelta(hours=dt) if m_h_d=='h' else timedelta(days=dt) if m_h_d == 'd' else None
				end_time = int((datetime.fromtimestamp(df[df.columns[0]].iloc[0]/1000)-timedt).timestamp()*1000)

				params = f'?&symbol={pair}&interval={timeframe}&limit={jeu}&endTime={end_time}'
				data   = self._get(self.base + self.endpoints['klines'] + params)
				df2    = pd.DataFrame.from_dict(data)
				df     = df2.append(df, ignore_index=True)		# Append rows of df to the end of df2

		# Clean-up
		df = df.drop(range(6, 12), axis=1)
		df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']

		for col in list(df.columns)[1:]:
			df[col] = df[col].astype(float)

		# Convert timestamp in milliseconds to datetime
		df['time'] = pd.to_datetime(df['time'] * 1000000, infer_datetime_format=True)

		return df


	def GetLastestPriceOfPair(self, pair:str)->str:
		""" Gets the last price for a pair """

		data = self._get(self.base + self.endpoints["lastestPrice"] + '?&symbol=' + pair)

		if data.__contains__('code'):
			return "In 'Exchange' : Could not get last price of " + pair + " from Binance."
		else:
			return data['price']		# "4.00000200"


	def Get24hrTicker(self, pair:str)->dict:
		""" Used to get the 24H rolling volume in quote for a pair, to sort the trading pairs."""

		url = self.base + self.endpoints['24hrTicker'] + "?symbol=" + pair
		return self._get(url)		# -> {'symbol': 'QTUMETH', 'priceChange': '0.00006700', ..., 'volume' = 37680, ...}


	def GetOrderBook(self, pair:str, limit:int)->dict:
		""" Gets the lastest orderbook for a pair. Max limit=5000"""

		url  = self.base + self.endpoints["orderBook"] + '?&symbol=' + pair + "&limit=" + str(limit)
		data = self._get(url)

		if data.__contains__('code'):
			print("In 'Exchange' : Could not get the order book of " + pair + " from Binance.")
			print(data)
			return {}
		else:
			return data


	def PlaceOrder(self, order_params:dict, test_order:bool):
		""" Places an order on Binance. """

		# Mandatory parameter to send to the endpoint:
		order_params['timestamp']  = self.GetServerTime()

		# Additional parameter to send to the endpoint:
		order_params['recvWindow']       = 5000
		order_params['newOrderRespType'] = 'RESULT'

		self.signRequest(order_params)

		if test_order:
			url = self.base + self.endpoints['testOrder']
		else:
			url = self.base + self.endpoints['order']

		order_result = self._post(url, params=order_params, headers=self.headers)

		return order_result


	def CancelOrder(self, pair:str, orderId:str):
		""" Cancels the order on a pair based on orderId. """

		params = {'symbol'	   : pair,
				  'orderId'    : orderId,
				  'recvWindow' : 5000,
				  'timestamp'  : self.GetServerTime()}

		self.signRequest(params)

		url = self.base + self.endpoints['order']

		try:
			response = requests.delete(url, params=params, headers=self.headers)
			data     = response.text
		except Exception as e:
			print("Exception occurred when trying to cancel order on "+url)
			print(e)
			data = {'code': '-1', 'msg':e}

		return json.loads(data)


	def GetOrderInfo(self, pair:str, orderId:str):
		""" Gets info about an order on a symbol based on orderId. """

		params = {'symbol'			  : pair,
				  'origClientOrderId' : orderId,
				  'recvWindow'		  : 5000,
				  'timestamp'		  : self.GetServerTime()}

		self.signRequest(params)

		url = self.base + self.endpoints['order']

		return self._get(url, params=params, headers=self.headers)


	def GetAllOrderInfo(self, pair:str):
		""" Gets info about all the orders on a symbol """

		params = {'symbol'   : pair,
				  'timestamp': self.GetServerTime()}

		self.signRequest(params)

		url = self.base + self.endpoints['allOrders']

		try:
			response = requests.get(url, params=params, headers=self.headers)
			data     = response.text
		except Exception as e:
			print("Exception occured when trying to get info on all orders on " + url)
			print(e)
			data = {'code': '-1', 'msg':e}

		return json.loads(data)


	def GetTimestamp(self):
		""" Returns the millisecond timestamp of when the request was created and sent.
			Takes into account the shift between the clocks. """

		# timestamp  = int(datetime.timestamp(datetime.now() + timedelta(milliseconds=self.TimeDifferenceWithBinance()))*1000)
		# timestamp  = int(time.time()*1000 + self.TimeDifferenceWithBinance())
		# timestamp  = self.GetServerTime()
		# serverTime = self.GetServerTime()
		#
		# recvWindow = 5000						# the number of milliseconds after timestamp the request is valid for.
		#
		# if timestamp < (serverTime + 1000) and (serverTime - timestamp) <= recvWindow:
		# 	return timestamp
		# else:
		# 	return 0

		diff = self.TimeDifferenceWithBinance()

		if diff > 0:
			# Binance ahead of us
			return self.GetServerTime() + 200
		else:
			# Binance behind us
			return int(time.time()*1000)


	def RoundToValidPrice(self, pair:str, price:Decimal)->Decimal:
		""" Addresses the issue of PRICE_FILTER """
		# https://github.com/binance-exchange/binance-official-api-docs/blob/master/rest-api.md#price_filter

		""" The price/stopPrice must obey 3 rules :
				price >= minPrice
				price <= maxPrice
				(price-minPrice) % tickSize == 0 
		"""

		pair_info = self.GetMetadataOfPair(pair=pair)
		pr_filter = {}

		for fil in pair_info["filters"]:
			if fil["filterType"] == "PRICE_FILTER":
				pr_filter = fil
				break

		if not pr_filter.keys().__contains__("tickSize"):
			raise Exception("Couldn't find tickSize or PRICE_FILTER in symbol_data.")

		""" pr_filter = {"filterType" : "PRICE_FILTER",
		 			     "minPrice"   : "0.00000100",
		 			     "maxPrice"   : "100000.00000000",
		 			     "tickSize"   : "0.00000100"}
		"""
		minPrice = Decimal(pr_filter['minPrice'])
		maxPrice = Decimal(pr_filter['maxPrice'])
		tickSize = Decimal(pr_filter['tickSize'])


		if minPrice <= price <= maxPrice:
			if (price-minPrice) % tickSize == 0:
				# Round down to the nearest tickSize and remove zeros after
				newPrice = price // tickSize * tickSize
				# print("Attention on {pair} : the price {price} is incorrect and has been rounded to {newPrice}".format(pair=pair, price=price, newPrice=newPrice))
				return newPrice
			else:
				return price

		if price < minPrice:
			# print("Attention on {pair} : the price {price} is too low and has been set to {minPrice}".format(pair=pair, price=price, minPrice=minPrice))
			return minPrice

		if price > maxPrice:
			# print("Attention on {pair} : the price {price} is too high and has been set to {maxPrice}".format(pair=pair, price=price, maxPrice=maxPrice))
			return maxPrice


	def RoundToValidQuantity(self, pair:str, quantity:Decimal)->Decimal:
		""" Addresses the issue of LOT_SIZE """
		# https://github.com/binance-exchange/binance-official-api-docs/blob/master/rest-api.md#lot_size

		""" The price/stopPrice must obey 3 rules :
				quantity >= minQty
				quantity <= maxQty
				(quantity-minQty) % stepSize == 0
		"""
		pair_info = self.GetMetadataOfPair(pair=pair)
		pr_filter = {}

		for fil in pair_info["filters"]:
			if fil["filterType"] == "LOT_SIZE":
				pr_filter = fil
				break

		if not pr_filter.keys().__contains__("stepSize"):
			raise Exception("Couldn't find stepSize or LOT_SIZE in symbol_data.")

		""" pr_filter = {"filterType" : "LOT_SIZE",
		 			     "minQty"     : "0.00100000",
					     "maxQty"     : "100000.00000000",
					     "stepSize"   : "0.00100000"}
		"""

		minQty   = Decimal(pr_filter['minQty'])
		maxQty   = Decimal(pr_filter['maxQty'])
		stepSize = Decimal(pr_filter['stepSize'])

		if minQty <= quantity <= maxQty:
			if (quantity-minQty) % stepSize != 0:
				# Round down to the nearest stepSize and remove zeros after
				newQuantity = quantity // stepSize * stepSize
				return newQuantity
			else:
				return quantity

		if quantity < minQty:
			return minQty

		if quantity > maxQty:
			return maxQty