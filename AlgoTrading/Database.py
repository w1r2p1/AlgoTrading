import sqlite3
from decimal import Decimal
from datetime import datetime


class BotDatabase:

	def __init__(self, name:str):
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
			quote 					text,
			timeframe 				text,
			status					text,
			quote_allocation		text,
			base_balance			text,
			quote_lockedintrade     text,
			base_value_in_quote		text,
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
            MinNotional				text,
            minPrice                text,
		    maxPrice                text,
		    tickSize                text,
		    minQty                  text,
		    maxQty                  text,
		    stepSize                text
			)''')

		c.execute('''CREATE TABLE IF NOT EXISTS orders (
			orderId 				text primary key,
			pair 					text,
			quote					text,
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
			timeInForce 			bool,
			liquidate_position		bool
			)''')

		c.execute('''CREATE TABLE IF NOT EXISTS account_balances (
			quote 							text,
			started_with					text,
			real_balance 					text,
			real_locked 					text,
			internal_balance				text,
			internal_locked					text,
			internal_profit					text,
			internal_quote_fees             text,
			internal_BNB_fees				text,
			internal_profit_minus_fees		text
			)''')

		conn.commit()


	""" BOTS --------------------------------------------------------------- """
	def SaveBot(self, bot_params:dict):

		conn 			 = sqlite3.connect(self.name, detect_types=sqlite3.PARSE_DECLTYPES)
		conn.row_factory = sqlite3.Row
		c 				 = conn.cursor()

		values = tuple(param for param in bot_params.values())

		c.execute('INSERT INTO bots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', values)
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
														# [{pair:'LTCBTC', 'is_active'=True, ...}, {pair:'ETHBTC, 'is_active'=True, ...}]

		except Exception as e:
			print(e)
			return False


	def update_bot(self, pair:str, **kwargs):
		""" Updates a Bot within the Database. """

		conn 			 = sqlite3.connect(self.name, detect_types=sqlite3.PARSE_DECLTYPES)
		conn.row_factory = sqlite3.Row
		c 				 = conn.cursor()

		if 'status' in kwargs:
			c.execute('UPDATE bots SET status = ? WHERE pair = ?', (kwargs['status'], pair))

		if 'quote_allocation' in kwargs:
			c.execute('UPDATE bots SET quote_allocation = ? WHERE pair = ?', (kwargs['quote_allocation'], pair))

		if 'base_balance' in kwargs:
			c.execute('UPDATE bots SET base_balance = ? WHERE pair = ?', (kwargs['base_balance'], pair))

		if 'quote_lockedintrade' in kwargs:
			c.execute('UPDATE bots SET quote_lockedintrade = ? WHERE pair = ?', (kwargs['quote_lockedintrade'], pair))

		if 'base_value_in_quote' in kwargs:
			c.execute('UPDATE bots SET base_value_in_quote = ? WHERE pair = ?', (kwargs['base_value_in_quote'], pair))

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


	def DeleteBots(self, quote:str):
		""" Deletes all the bots in the database on a given quote. """

		if quote:
			conn 			 = sqlite3.connect(self.name, detect_types=sqlite3.PARSE_DECLTYPES)
			conn.row_factory = sqlite3.Row
			c 				 = conn.cursor()

			# Check if bots exist on the quote
			c.execute('SELECT * FROM bots WHERE quote=?', (quote, ))
			details = c.fetchall()

			if details:
				for item in details:
					if 'Looking to exit' in dict(item)['status']:
						answer = input(f"You have open position(s) on {quote}, are you sure you want to delete its records ?")
						if answer == 'n':
							return
				c.execute('DELETE FROM bots WHERE quote=?', (quote, ))
				conn.commit()
				print(f"Deleted all records of {quote}.")



	""" ORDERS ------------------------------------------------------------- """
	def SaveOrder(self, quote:str, order_result:dict, hold_duration:str, profit:str, quote_fee:str, BNB_fee:str, profit_minus_fees:str, time_to_fill:str, **kwargs):
		""" Saves an order to the Database. """

		if order_result:
			conn 			 = sqlite3.connect(self.name, detect_types=sqlite3.PARSE_DECLTYPES)
			conn.row_factory = sqlite3.Row
			c 				 = conn.cursor()

			# values = tuple(param for param in order_result.values())
			values = (order_result['orderId'],
					  order_result['symbol'],
					  quote,
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
					  order_result['timeInForce'],
					  kwargs.get('liquidate_position', False))

			c.execute('INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', values)
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


	def get_quote_orders(self, quote:str):
		""" Gets all the orders we have made on a quote"""

		conn             = sqlite3.connect(self.name, detect_types=sqlite3.PARSE_DECLTYPES)
		conn.row_factory = sqlite3.Row
		c                = conn.cursor()

		c.execute('SELECT * FROM orders WHERE quote=?', (quote, ))
		orders = c.fetchall()
													# list(orders) = [<sqlite3.Row object at 0x0000020664E28670>, <sqlite3.Row object at 0x0000020664E9BA70>, ...]
		return orders                           	# We need to return None if there is no bot on the pair, so no dict(orders)


	def GetOpenBots(self, quote:str):
		""" Gets all the orders made on a pair """

		conn             = sqlite3.connect(self.name, detect_types=sqlite3.PARSE_DECLTYPES)
		conn.row_factory = sqlite3.Row
		c                = conn.cursor()

		c.execute('SELECT * FROM bots WHERE quote=? AND status = "Looking to exit"', (quote, ))
		openBots = c.fetchall()
													# list(openBots) = [<sqlite3.Row object at 0x0000020664E28670>, <sqlite3.Row object at 0x0000020664E9BA70>, ...]
													# dict(openBots) = {'pair': 'STORMETH', 'side': 'BUY',...}
		return openBots                           	# We need to return None if there is no opened bot , so no dict(openBots)




	""" BALANCES ------------------------------------------------------------ """
	def initiate_db_account_balances(self, quote:str, started_with:str, real_balance:str, internal_balance:str):
		""" Creates an account balance for a quote. """

		conn 			 = sqlite3.connect(self.name, detect_types=sqlite3.PARSE_DECLTYPES)
		conn.row_factory = sqlite3.Row
		c 				 = conn.cursor()

		values = (quote,
				  started_with,
				  real_balance,
				  '0',
				  internal_balance,
				  '0',
				  '0',
				  '0',
				  '0',
				  '0',
				  quote)
		# Insert a row for the quoreasset only if doesn't already exist
		c.execute('INSERT INTO account_balances SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?, ? WHERE NOT EXISTS (SELECT * FROM account_balances WHERE quote=?)', values)

		conn.commit()


	def update_db_account_balances(self,
								   quote:str,
								   real_balance:str,
								   real_locked:str,
								   internal_balance:str,
								   internal_locked:str,
								   internal_profit:str,
								   internal_quote_fees:str,
								   internal_BNB_fees:str,
								   internal_profit_minus_fees:str,
								   quoteAssetPrecision:int,
								   BNB_Precision:int):
		""" Updates the internal & real balances of each quote. """

		conn 			 = sqlite3.connect(self.name, detect_types=sqlite3.PARSE_DECLTYPES)
		conn.row_factory = sqlite3.Row
		c 				 = conn.cursor()

		# Get the current balance in the db
		c.execute('SELECT * FROM account_balances WHERE quote = ?', (quote, ))
		accountBalance 	 	  = c.fetchone()
		int_balance 		  = dict(accountBalance)['internal_balance']
		int_locked 		  	  = dict(accountBalance)['internal_locked']
		int_profit   		  = dict(accountBalance)['internal_profit']
		int_quote_fees 	  	  = dict(accountBalance)['internal_quote_fees']
		int_BNB_fees 		  = dict(accountBalance)['internal_BNB_fees']
		int_profit_minus_fees = dict(accountBalance)['internal_profit_minus_fees']

		values = (real_balance,
				  real_locked,
				  format(round(Decimal(int_balance) + Decimal(internal_balance), quoteAssetPrecision), 'f'),
				  format(round(Decimal(int_locked)  + Decimal(internal_locked),  quoteAssetPrecision), 'f'),
				  format(round(Decimal(int_profit)  + Decimal(internal_profit),  quoteAssetPrecision), 'f'),
				  format(round(Decimal(int_quote_fees) + Decimal(internal_quote_fees), quoteAssetPrecision), 'f'),
				  format(round(Decimal(int_BNB_fees)   + Decimal(internal_BNB_fees),   BNB_Precision), 'f'),
				  format(round(Decimal(int_profit_minus_fees) + Decimal(internal_profit_minus_fees), quoteAssetPrecision), 'f'),
				  quote)

		c.execute('UPDATE account_balances SET real_balance = ?,'
				  							  'real_locked = ?,'
										      'internal_balance = ?,'
										      'internal_locked = ?,'
										      'internal_profit = ?,'
										  	  'internal_quote_fees = ?,'
										 	  'internal_BNB_fees = ?,'
										 	  'internal_profit_minus_fees = ?'
				  'WHERE quote = ?', values)

		conn.commit()


	def get_db_account_balance(self, quote:str, **kwargs)->str:

		conn 			 = sqlite3.connect(self.name, detect_types=sqlite3.PARSE_DECLTYPES)
		conn.row_factory = sqlite3.Row
		c 				 = conn.cursor()

		c.execute('SELECT * FROM account_balances WHERE quote = ?', (quote, ))
		balance = c.fetchone()

		if kwargs.get('started_with', False):
			return dict(balance)['started_with']

		if kwargs.get('real_balance', False):
			return dict(balance)['real_balance']

		if kwargs.get('real_locked', False):
			return dict(balance)['real_locked']

		if kwargs.get('internal_balance', False):
			return dict(balance)['internal_balance']

		if kwargs.get('internal_locked', False):
			return dict(balance)['internal_locked']

		if kwargs.get('internal_profit', False):
			return dict(balance)['internal_profit']

		if kwargs.get('internal_quote_fees', False):
			return dict(balance)['internal_quote_fees']

		if kwargs.get('internal_BNB_fees', False):
			return dict(balance)['internal_BNB_fees']

		if kwargs.get('internal_profit_minus_fees', False):
			return dict(balance)['internal_profit_minus_fees']