import sqlite3
from decimal import Decimal
from datetime import datetime


class BotDatabase:

	def __init__(self, name:str):
		# sqlite3.register_adapter(Decimal, DecimalToString)          	# Decimal type converted to string before being written to the db.
		# sqlite3.register_converter("decimal", convert_to_decimal)		# bytestring from the database into a custom Python type
		self.name = name
		self.initialise()


	def initialise(self):
		""" Initialises the Database """

		conn 			 = sqlite3.connect(self.name, detect_types=sqlite3.PARSE_DECLTYPES)
		conn.row_factory = sqlite3.Row
		c 				 = conn.cursor()

		# Create tables
		c.execute('''CREATE TABLE IF NOT EXISTS bots (
			pair					text primary key,
			quoteasset 				text,
			timeframe 				text,
			status					text,
			quoteBalance			text,
			baseBalance				text,
			last_order_date			text,
			last_profit 			text,
			bot_profit	 			text,
			bot_quote_fees			text,
			bot_BNB_fees			text,
			bot_profit_minus_fees	text,
			profitable_sells 		int,
			unprofitable_sells		int,
			number_of_orders		int,
			utc_creation_date		text,
			quoteAssetPrecision  	int,
            baseAssetPrecision   	int,
            quotePrecision      	int,
            BNB_precision			int,
            MinNotional				text
			)''')

		c.execute('''CREATE TABLE IF NOT EXISTS orders (
			orderId 				text primary key,
			pair 					text,
			quoteasset				text,
			side 					text,
			order_type				text,
			status 					text,
			orderListId 			text,
			clientOrderId 			text,
			transactTime 			text,
			price 					text,
			origQty 				text,
			executedQty 			text,
			cummulativeQuoteQty		text,
			profit					text,
			quote_fee				text,
			BNB_fee					text,
			profit_minus_fees		text,
			time_to_fill			text,
			hold_duration			text,
			timeInForce 			bool 
			)''')

		c.execute('''CREATE TABLE IF NOT EXISTS account_balances (
			quoteasset 						text,
			started_with					text,
			real_quote_balance 				text,
			real_profit						text,
			internal_quote_balance			text,
			internal_profit					text,
			internal_quote_fees             text,
			internal_BNB_fees				text,
			internal_profit_minus_fees		text
			)''')

		conn.commit()


	""" BOTS """
	def SaveBot(self, bot_params:dict):

		conn 			 = sqlite3.connect(self.name, detect_types=sqlite3.PARSE_DECLTYPES)
		conn.row_factory = sqlite3.Row
		c 				 = conn.cursor()

		values = tuple(param for param in bot_params.values())

		c.execute('INSERT INTO bots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', values)
		conn.commit()


	def GetBot(self, pair:str):
		""" Gets Bot details from Database. """

		conn 			 = sqlite3.connect(self.name, detect_types=sqlite3.PARSE_DECLTYPES)
		conn.row_factory = sqlite3.Row
		c 				 = conn.cursor()

		c.execute('SELECT * FROM bots WHERE pair = ?', (pair, ))
		bot = c.fetchone()
		return bot					# We need to return None if there is no bot on the pair, so no dict(bot). 		# dict(bot) = {botname:'bot_LTCBTC', pair:'LTCBTC, ...}


	def GetAllBots(self):
		""" Gets details from all the bots in the database """
		try:
			conn 			 = sqlite3.connect(self.name, detect_types=sqlite3.PARSE_DECLTYPES)
			conn.row_factory = sqlite3.Row
			c 				 = conn.cursor()
			c.execute('SELECT * FROM bots')
			all_bots = c.fetchall()
			return all_bots								# list(all_bots) = [<sqlite3.Row object at 0x000001BB27302FD0>, <sqlite3.Row object at 0x000001BB27302CB0>,...]
														# [{botname:'bot_LTCBTC', pair:'LTCBTC, 'is_active'=True, ...}, {botname:'bot_ETHBTC', pair:'ETHBTC, 'is_active'=True, ...}]

		except Exception as e:
			print(e)
			return False


	def UpdateBot(self, pair:str, **kwargs):
		""" Updates a Bot within the Database. """

		conn 			 = sqlite3.connect(self.name, detect_types=sqlite3.PARSE_DECLTYPES)
		conn.row_factory = sqlite3.Row
		c 				 = conn.cursor()

		if 'status' in kwargs:
			c.execute('UPDATE bots SET status = ? WHERE pair = ?', (kwargs['status'], pair))

		if 'quoteBalance' in kwargs:
			c.execute('UPDATE bots SET quoteBalance = ? WHERE pair = ?', (kwargs['quoteBalance'], pair))

		if 'baseBalance' in kwargs:
			c.execute('UPDATE bots SET baseBalance = ? WHERE pair = ?', (kwargs['baseBalance'], pair))

		if 'last_order_date' in kwargs:
			c.execute('UPDATE bots SET last_order_date = ? WHERE pair = ?', (kwargs['last_order_date'], pair))

		if 'last_profit' in kwargs:
			c.execute('UPDATE bots SET last_profit = ? WHERE pair = ?', (kwargs['last_profit'], pair))

		if 'profitable_sells' in kwargs:
			c.execute('UPDATE bots SET profitable_sells = profitable_sells + ? WHERE pair = ?', (kwargs['profitable_sells'], pair))

		if 'unprofitable_sells' in kwargs:
			c.execute('UPDATE bots SET unprofitable_sells = unprofitable_sells + ? WHERE pair = ?', (kwargs['unprofitable_sells'], pair))

		if 'number_of_orders' in kwargs:
			c.execute('UPDATE bots SET number_of_orders = number_of_orders + ? WHERE pair = ?', (kwargs['number_of_orders'], pair))

		if 'bot_profit' in kwargs:
			c.execute('SELECT * FROM bots WHERE pair = ?', (pair, ))
			ov = dict(c.fetchone())['bot_profit']
			values = (str(Decimal(ov) + Decimal(kwargs['bot_profit'])), pair)
			c.execute('UPDATE bots SET bot_profit = ? WHERE pair = ?', values)

		if 'bot_quote_fees' in kwargs:
			c.execute('SELECT * FROM bots WHERE pair = ?', (pair, ))
			ov = dict(c.fetchone())['bot_quote_fees']
			values = (str(Decimal(ov) + Decimal(kwargs['bot_quote_fees'])), pair)
			c.execute('UPDATE bots SET bot_quote_fees = ? WHERE pair = ?', values)

		if 'bot_BNB_fees' in kwargs:
			c.execute('SELECT * FROM bots WHERE pair = ?', (pair, ))
			ov = dict(c.fetchone())['bot_BNB_fees']
			values = (str(Decimal(ov) + Decimal(kwargs['bot_BNB_fees'])), pair)
			c.execute('UPDATE bots SET bot_BNB_fees = ? WHERE pair = ?', values)

		if 'bot_profit_minus_fees' in kwargs:
			c.execute('SELECT * FROM bots WHERE pair = ?', (pair, ))
			ov = dict(c.fetchone())['bot_profit_minus_fees']
			values = (str(Decimal(ov) + Decimal(kwargs['bot_profit_minus_fees'])), pair)
			c.execute('UPDATE bots SET bot_profit_minus_fees = ? WHERE pair = ?', values)

		conn.commit()


	def DeleteBots(self, quoteasset:str):
		""" Deletes all the bots in the database on a given quoteasset. """

		if quoteasset:
			conn 			 = sqlite3.connect(self.name, detect_types=sqlite3.PARSE_DECLTYPES)
			conn.row_factory = sqlite3.Row
			c 				 = conn.cursor()

			# Check if bots exist on the quoteasset
			c.execute('SELECT * FROM bots WHERE quoteasset=?', (quoteasset, ))
			details = c.fetchall()

			if details:
				for item in details:
					if 'Looking to exit' in dict(item)['status']:
						answer = input("You have open position(s) on " + quoteasset + ", are you sure you want to delete its records ?")
						if answer == 'n':
							return
				c.execute('DELETE FROM bots WHERE quoteasset=?', (quoteasset, ))
				conn.commit()
				print("Deleted all records of " + quoteasset + ".")



	""" ORDERS """
	def SaveOrder(self, quoteasset:str, order_result:dict, hold_duration:str, profit:str, quote_fee:str, BNB_fee:str, profit_minus_fees:str, time_to_fill:str):
		""" Saves an order to the Database. """

		if order_result:
			conn 			 = sqlite3.connect(self.name, detect_types=sqlite3.PARSE_DECLTYPES)
			conn.row_factory = sqlite3.Row
			c 				 = conn.cursor()

			# values = tuple(param for param in order_result.values())
			values = (order_result['orderId'],
					  order_result['symbol'],
					  quoteasset,
					  order_result['side'],
					  order_result['type'],
					  order_result['status'],
					  order_result['orderListId'],
					  order_result['clientOrderId'],
					  datetime.utcfromtimestamp(int(order_result['transactTime'])/1000).strftime("%Y-%m-%d %H:%M:%S"),
					  order_result['price'],
					  order_result['origQty'],
					  order_result['executedQty'],
					  order_result['cummulativeQuoteQty'],
					  profit,
					  quote_fee,
					  BNB_fee,
					  profit_minus_fees,
					  time_to_fill,
					  hold_duration,
					  order_result['timeInForce'])

			c.execute('INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', values)
			conn.commit()


	def GetOrdersOfBot(self, pair:str):
		""" Gets all the orders made on a pair """

		conn             = sqlite3.connect(self.name, detect_types=sqlite3.PARSE_DECLTYPES)
		conn.row_factory = sqlite3.Row
		c                = conn.cursor()

		c.execute('SELECT * FROM orders WHERE pair=?', (pair, ))
		orders = c.fetchall()
													# list(orders) = [<sqlite3.Row object at 0x0000020664E28670>, <sqlite3.Row object at 0x0000020664E9BA70>, ...]
													# dict(order)  = {'pair': 'STORMETH', 'side': 'BUY',...}
		return orders                           	# We need to return None if there is no bot on the pair, so no dict(orders)


	def get_quote_orders(self, quoteasset:str):
		""" Gets all the orders we have made on a quoteasset"""

		conn             = sqlite3.connect(self.name, detect_types=sqlite3.PARSE_DECLTYPES)
		conn.row_factory = sqlite3.Row
		c                = conn.cursor()

		c.execute('SELECT * FROM orders WHERE quoteasset=?', (quoteasset, ))
		orders = c.fetchall()
													# list(orders) = [<sqlite3.Row object at 0x0000020664E28670>, <sqlite3.Row object at 0x0000020664E9BA70>, ...]
		return orders                           	# We need to return None if there is no bot on the pair, so no dict(orders)


	def GetOpenBots(self, quoteasset:str):
		""" Gets all the orders made on a pair """

		conn             = sqlite3.connect(self.name, detect_types=sqlite3.PARSE_DECLTYPES)
		conn.row_factory = sqlite3.Row
		c                = conn.cursor()

		c.execute('SELECT * FROM bots WHERE quoteasset=? AND status = "Looking to exit"', (quoteasset, ))
		openBots = c.fetchall()
													# list(openBots) = [<sqlite3.Row object at 0x0000020664E28670>, <sqlite3.Row object at 0x0000020664E9BA70>, ...]
													# dict(openBots) = {'pair': 'STORMETH', 'side': 'BUY',...}
		return openBots                           	# We need to return None if there is no opened bot , so no dict(openBots)




	""" BALANCES """
	def InitiateAccountBalance(self, quoteasset:str, started_with:str, real_quote_balance:str, internal_quote_balance:str):
		""" Creates an account balance for a quoteasset. """

		conn 			 = sqlite3.connect(self.name, detect_types=sqlite3.PARSE_DECLTYPES)
		conn.row_factory = sqlite3.Row
		c 				 = conn.cursor()

		values = (quoteasset,
				  started_with,
				  real_quote_balance,
				  '0',
				  internal_quote_balance,
				  '0',
				  '0',
				  '0',
				  '0',
				  quoteasset)
		# Insert a row for the quoreasset only if doesn't already exist
		c.execute('INSERT INTO account_balances SELECT ?, ?, ?, ?, ?, ?, ?, ?, ? WHERE NOT EXISTS (SELECT * FROM account_balances WHERE quoteasset=?)', values)

		conn.commit()


	def UpdateAccountBalance(self,
							 quoteasset:str,
							 real_quote_balance:str,
							 real_profit:str,
							 internal_quote_balance:str,
							 internal_profit:str,
							 internal_quote_fees:str,
							 internal_BNB_fees:str,
							 internal_profit_minus_fees:str,
							 quoteAssetPrecision:int,
							 BNB_Precision:int):
		""" Updates the internal & real balances of each quoteasset. """

		conn 			 = sqlite3.connect(self.name, detect_types=sqlite3.PARSE_DECLTYPES)
		conn.row_factory = sqlite3.Row
		c 				 = conn.cursor()

		# Get the current balance in the db
		c.execute('SELECT * FROM account_balances WHERE quoteasset = ?', (quoteasset, ))
		accountBalance 	 	  = c.fetchone()
		int_balance 		  = dict(accountBalance)['internal_quote_balance']
		int_profit   		  = dict(accountBalance)['internal_profit']
		int_quote_fees 	  	  = dict(accountBalance)['internal_quote_fees']
		int_BNB_fees 		  = dict(accountBalance)['internal_BNB_fees']
		int_profit_minus_fees = dict(accountBalance)['internal_profit_minus_fees']

		values = (real_quote_balance,
				  real_profit,
				  format(round(Decimal(int_balance) + Decimal(internal_quote_balance), quoteAssetPrecision), 'f'),
				  format(round(Decimal(int_profit) + Decimal(internal_profit),         quoteAssetPrecision), 'f'),
				  format(round(Decimal(int_quote_fees) + Decimal(internal_quote_fees), quoteAssetPrecision), 'f'),
				  format(round(Decimal(int_BNB_fees)   + Decimal(internal_BNB_fees),   BNB_Precision),       'f'),
				  format(round(Decimal(int_profit_minus_fees) + Decimal(internal_profit_minus_fees), quoteAssetPrecision), 'f'),
				  quoteasset)

		c.execute('UPDATE account_balances SET real_quote_balance = ?,'
				  							  'real_profit = ?,'
										      'internal_quote_balance = ?,'
										      'internal_profit = ?,'
										  	  'internal_quote_fees = ?,'
										 	  'internal_BNB_fees = ?,'
										 	  'internal_profit_minus_fees = ?'
				  'WHERE quoteasset = ?', values)

		conn.commit()


	def GetAccountBalance(self, quoteasset:str, real_or_internal:str)->str:

		conn 			 = sqlite3.connect(self.name, detect_types=sqlite3.PARSE_DECLTYPES)
		conn.row_factory = sqlite3.Row
		c 				 = conn.cursor()

		c.execute('SELECT * FROM account_balances WHERE quoteasset = ?', (quoteasset, ))
		balance = c.fetchone()

		if real_or_internal == 'real':
			return dict(balance)['real_quote_balance']				# '2.50000000'
		elif real_or_internal == 'internal':
			return dict(balance)['internal_quote_balance']			# '2.50000000'


	def get_profit(self, quoteasset:str, real_or_internal:str)->str:

		conn 			 = sqlite3.connect(self.name, detect_types=sqlite3.PARSE_DECLTYPES)
		conn.row_factory = sqlite3.Row
		c 				 = conn.cursor()

		c.execute('SELECT * FROM account_balances WHERE quoteasset = ?', (quoteasset, ))
		balance = c.fetchone()

		return dict(balance)[f'{real_or_internal}_profit']				# '2.50000000'


	def GetQuoteFees(self, quoteasset:str)->str:

		conn 			 = sqlite3.connect(self.name, detect_types=sqlite3.PARSE_DECLTYPES)
		conn.row_factory = sqlite3.Row
		c 				 = conn.cursor()

		c.execute('SELECT * FROM account_balances WHERE quoteasset = ?', (quoteasset, ))
		balance = c.fetchone()

		return dict(balance)['internal_quote_fees']				# '2.50000000'


	def GetBNBFees(self, quoteasset:str)->str:

		conn 			 = sqlite3.connect(self.name, detect_types=sqlite3.PARSE_DECLTYPES)
		conn.row_factory = sqlite3.Row
		c 				 = conn.cursor()

		c.execute('SELECT * FROM account_balances WHERE quoteasset = ?', (quoteasset, ))
		balance = c.fetchone()

		return dict(balance)['internal_BNB_fees']				# '2.50000000'


	def GetProfit_minus_fees(self, quoteasset:str)->str:

		conn 			 = sqlite3.connect(self.name, detect_types=sqlite3.PARSE_DECLTYPES)
		conn.row_factory = sqlite3.Row
		c 				 = conn.cursor()

		c.execute('SELECT * FROM account_balances WHERE quoteasset = ?', (quoteasset, ))
		balance = c.fetchone()

		return dict(balance)['internal_profit_minus_fees']				# '2.50000000'


	def GetStartBalance(self, quoteasset:str)->str:

		conn 			 = sqlite3.connect(self.name, detect_types=sqlite3.PARSE_DECLTYPES)
		conn.row_factory = sqlite3.Row
		c 				 = conn.cursor()

		c.execute('SELECT * FROM account_balances WHERE quoteasset = ?', (quoteasset, ))
		balance = c.fetchone()

		return dict(balance)['started_with']				# '2.50000000'