from   Exchange import Binance
from   Database import BotDatabase

from   uuid     import uuid1
from   yaspin   import yaspin
from   multiprocessing.pool import ThreadPool as Pool
from   functools import partial
import random
from   itertools import islice
import sys
import colorama
import configparser as cfg
import requests
import pause
import time
import datetime
from   datetime import datetime, timedelta
from   decimal import Decimal
import re
import os

# Clear file database.db
with open('assets/database.db', "w"):
    pass

class Strategy:

    def __init__(self, name:str):
        self.name = name

    @staticmethod
    def find_signal(df)->str:
        """ Compute the indicators and look for a signal.
            Each strategy is only a template that needs to be fine tuned.
        """

        strategy_ = 'Crossover'

        if strategy_=='Crossover':
            indic = 'ssf'
            fast_length = 40*24
            slow_length = 5*24

            fast_name = f'{indic}_{fast_length}'
            slow_name = f'{indic}_{slow_length}'
            getattr(df.ta, indic)(close=df.loc[:,'close'], length=fast_length, append=True, col_names=(fast_name,))
            getattr(df.ta, indic)(close=df.loc[:,'close'], length=slow_length, append=True, col_names=(slow_name,))

            i = len(df.loc[:, 'close'])-1
            fast         = df[fast_name].iloc[i]
            slow         = df[slow_name].iloc[i]
            fast_shifted = df[fast_name].iloc[i-1]
            slow_shifted = df[slow_name].iloc[i-1]

            return 'sell' if fast_shifted > slow_shifted and fast < slow else 'buy' if fast_shifted < slow_shifted and fast > slow else ''


exchange = Binance(filename='assets/credentials.txt')
# database = BotDatabase(name="/assets/database.db")
database = BotDatabase(name=os.path.join(os.getcwd(), 'assets', 'database.db'))             # name =  'C:\Users\grego\Documents\GitHub\AlgoTrading\AlgoTrading\assets\database.db'
# database = BotDatabase(name=f"r{os.path.join(os.getcwd(), 'assets', 'database.db')}")       # name = r'C:\Users\grego\Documents\GitHub\AlgoTrading\AlgoTrading\assets\database.db'

strategy = Strategy(name='SSF_Crossover')


class Trading:

    def __init__(self, paper_trading:bool=True):
        self.paper_trading = paper_trading

        # List of all the quoteassets present in the database
        self.existing_quoteassets = list(set([dict(bot)['quoteasset'] for bot in database.GetAllBots()]))             # ['ETH', 'BTC']

        # For the Telegram connexion
        parser = cfg.ConfigParser()
        parser.read("assets/telegram_config.cfg")
        self.bot_token  = parser.get('creds', 'token')
        self.bot_chatID = '456212622'


    def main_loop(self):

        def create_bots():
            """ Creates all the possible bots on each quote if they don't exist. """

            sp       = yaspin()
            sp.color = 'green'

            def create_bot(pair_:str, quote_:str, timeframe_:str):
                """ In the database, create and save a bot with a set of parameters. """

                # Get the quotePrecision for this pair
                pair_info = exchange.GetMetadataOfPair(pair=pair_)
                BNB_info  = exchange.GetMetadataOfPair(pair='BNB'+quote_)

                if pair_info:
                    # Create the bot with a set of parameters
                    bot = dict(pair				     = pair_,
                               quoteasset		     = quote_,
                               timeframe 		     = timeframe_,
                               status                = '',
                               quoteBalance          = '',
                               baseBalance           = '',
                               last_order_date       = '',
                               last_profit		     = '',
                               bot_profit 	         = '0',
                               bot_quote_fees        = '0',
                               bot_BNB_fees          = '0',
                               bot_profit_minus_fees = '0',
                               profitable_sells      = 0,
                               unprofitable_sells    = 0,
                               number_of_orders      = 0,
                               utc_creation_date     = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                               quoteAssetPrecision   = int(pair_info['quoteAssetPrecision']),
                               baseAssetPrecision    = int(pair_info['baseAssetPrecision']),
                               quotePrecision        = int(pair_info['quotePrecision']),
                               BNB_precision         = int(BNB_info['baseAssetPrecision']),)

                    database.SaveBot(bot)			# Each row of the table 'bots' is made of the values of 'bot'.

            # List of all the quoteassets present in the database
            existing_quoteassets = set([dict(bot)['quoteasset'] for bot in database.GetAllBots()])                              # ['ETH', 'BTC']
            # Unique values of the two. | = union of 2 sets.
            AllQuoteassets = list(existing_quoteassets | set(WantedQuoteAssets))                                                # ['ETH', 'BTC']

            pairs  = exchange.GetNameOfPairs_WithQuoteasset(AllQuoteassets)                                                     # {'ETH': ['QTUMETH', 'EOSETH',..], 'BTC':[]}

            if pairs == {}:
                sys.exit("Could not get the pairs from Binance to create the bots. Exiting the script.")

            for quote_ in AllQuoteassets:
                sp.text = f"Creating the bots on {quote_}"
                sp.start()
                # Create the bots that don't already exist in parallel. Use the default number of workers in Pool(), which given by os.cpu_count(). Here, 8 are used.
                pool_  = Pool()
                func_bot = partial(create_bot, quote_=quote_, timeframe_=timeframe)
                pairs_without_bot = [pair for pair in pairs[quote_] if not database.GetBot(pair)]
                pool_.map(func_bot, pairs_without_bot)
                pool_.close()
                pool_.join()
                sp.stop()
                if quote_ in WantedQuoteAssets and quote_ in existing_quoteassets:
                    print(f"You want to trade on {quote_}, which is already in the database. {len(pairs_without_bot)} new pairs available. Total = {len(pairs[quote_])}.")
                elif quote_ in WantedQuoteAssets:
                    print(f"You want to trade on {quote_}, which is not in the database. Created {len(pairs_without_bot)} bots.")
                else:
                    print(f"Note that {quote_} is already existing in the database. {len(pairs_without_bot)} new pairs available. Total = {len(pairs[quote_])}.")

        create_bots()

        # In the 'account_balances' db, create an a line for each quoteasset (if they not already existing)
        for quote in self.existing_quoteassets:
            database.InitiateAccountBalance(quoteasset             = quote,
                                            started_with           = exchange.GetAccountBalance(quote) if not self.paper_trading else 1, # if paper trading, start with a balance of 1 for each quoteasset
                                            real_quote_balance     = exchange.GetAccountBalance(quote),
                                            internal_quote_balance = exchange.GetAccountBalance(quote) if not self.paper_trading else 1)

        colorama.init()
        print("\n_______________________________________________________________________________________________")
        print(colorama.Fore.GREEN + f"PAPER TRADING IS LIVE.\nTimeframe : {timeframe}.\n")

        while True:

            # Pause the program until the next candle arrives. Just before the candle arrives, set the pairs to trade on.
            wait = self.wait_for_candle()

            if not wait:
                # When the candle appears, start scanning
                print("\nTrading with a clocks difference of {time} ms.".format(time=exchange.TimeDifferenceWithBinance()))

                buys  = dict()
                pool  = Pool()
                func1 = partial(self.buy_order, dict_to_fill=buys)
                pool.map(func1, [dict(bot) for bot in database.GetAllBots() if dict(bot)['status']=='Looking to enter'])
                pool.close()
                pool.join()
                print("\tChecked all pairs for buy  signals.")

                sells = dict()
                pool  = Pool()
                func2 = partial(self.sell_order, dict_to_fill=sells)
                pool.map(func2, [dict(bot) for bot in database.GetAllBots() if dict(bot)['status']=='Looking to exit'])
                pool.close()
                pool.join()
                print("\tChecked all pairs for sell signals.")

                # Summary of the search
                self.print_trades_sequence(buys=buys, sells=sells)
                # break


    def buy_order(self, bot:dict, dict_to_fill:dict):
        """ The bot looks for a buy signal. If yes, places an order. """

        pair 	   = bot['pair']
        sp         = yaspin()
        sp.color   = 'red'
        sp.text    = "\tChecking for buy signals on " + pair + ", timeframe " + bot['timeframe'] + "."
        df         = exchange.GetPairKlines(pair=pair, timeframe=bot['timeframe'], candles=200)				                    # get dataframe

        if df.empty:
            return None

        # signal = strategy.find_signal(df=df)		# check for a signal (returns 'buy'/'sell')
        signal = random.choice(['buy', 'sell'])

        if signal == 'buy':

            # Mandatory parameters to send to the endpoint:
            buy_order_parameters = dict(symbol = pair,
                                        side   = "BUY",
                                        type   = "LIMIT")

            quoteOrderQty, buy_price, quantity = Decimal('0'), Decimal('0'), Decimal('0')

            # Additional mandatory parameters based on type
            if buy_order_parameters['type'] == 'MARKET':
                quoteOrderQty = Decimal(database.GetBot(pair)['quoteBalance'])
                buy_order_parameters['quoteOrderQty'] = format(round(quoteOrderQty, bot['quoteAssetPrecision']), 'f')		    # specifies the amount the user wants to spend (when buying) or receive (when selling) of the quote asset; the correct quantity will be determined based on the market liquidity and quoteOrderQty

            elif buy_order_parameters['type'] == 'LIMIT':
                market_price  = exchange.GetLastestPriceOfPair(pair=pair)                                                       # market_price is a string
                buy_price     = exchange.RoundToValidPrice(pair=pair, price=Decimal(market_price)*Decimal(0.99))                # buy_price is a Decimal        # Addresses the issue of PRICE_FILTER
                quantity      = exchange.RoundToValidQuantity(pair=pair, quantity=Decimal(bot['quoteBalance'])/buy_price)       # quantity  is a Decimal        # Addresses the issue of LOT_SIZE
                quoteOrderQty = buy_price*quantity

                buy_order_parameters['timeInForce'] = 'GTC'		 		                                                        # 'GTC' (Good-Till-Canceled), 'IOC' (Immediate-or-Cancel) (part or all of the order) or 'FOK' (Fill-or-Kill) (whole order)
                buy_order_parameters['price'] 	    = format(round(buy_price, bot['baseAssetPrecision']), 'f')
                buy_order_parameters['quantity']    = format(round(quantity,  bot['baseAssetPrecision']), 'f')

            # Place a buy order
            place_order_time = datetime.utcnow()
            buy_order_result = exchange.PlaceOrder(order_params=buy_order_parameters, test_order=self.paper_trading)

            if self.paper_trading:
                dummy_buy_order_result = {"symbol"               : pair,
                                          "orderId"              : str(uuid1()),
                                          "orderListId"          : -1,
                                          "clientOrderId"        : "Buy_6gCrw2kRUAF9CvJDGP16IP",
                                          "transactTime"         : str(exchange.GetServerTime()),
                                          "price"                : format(round(buy_price,     bot['quotePrecision']),      'f'),
                                          "origQty"              : format(round(quantity,      bot['baseAssetPrecision']),  'f'),
                                          "executedQty"          : format(round(quantity,      bot['baseAssetPrecision']),  'f'),
                                          "cummulativeQuoteQty"  : format(round(quoteOrderQty, bot['quoteAssetPrecision']), 'f'),     # On suppose que tout est passé et que les frais ont été payés en BNB
                                          "status"               : "FILLED",
                                          "timeInForce"          : "GTC",
                                          "type"                 : buy_order_parameters['type'],
                                          "side"                 : buy_order_parameters['side']}
                buy_order_result = {**dummy_buy_order_result, **buy_order_result}

            if "code" in buy_order_result:
                formatted_transactTime = datetime.utcfromtimestamp(int(buy_order_result['transactTime'])/1000).strftime('%H:%M:%S')
                print(f"\t{formatted_transactTime} - Error in placing a buy {'test' if self.paper_trading else ''} order on {pair} at {buy_price} :/")
                database.UpdateBot(pair=pair, status='', quoteBalance='')
                print(pair, buy_order_result)
                return None

            else:
                text = "\t{time} - Success in placing a buy {test} order on {pair} \t: bought {quantity} {base} \tat {price} {quoteasset} for {quoteQty} {quoteasset}.".format(time       = datetime.utcfromtimestamp(int(buy_order_result['transactTime'])/1000).strftime("%H:%M:%S"),
                                                                                                                                                                               test       = 'test' if self.paper_trading else '',
                                                                                                                                                                               pair       = pair,
                                                                                                                                                                               quantity   = buy_order_result['executedQty'],
                                                                                                                                                                               base       = pair.replace(bot['quoteasset'], ''),
                                                                                                                                                                               price      = buy_price,
                                                                                                                                                                               quoteasset = bot['quoteasset'],
                                                                                                                                                                               quoteQty   = buy_order_result['cummulativeQuoteQty'])
                print(text)

                # Compute the time taken to fill the order
                timedelta_to_fill = datetime.utcfromtimestamp(int(buy_order_result['transactTime'])/1000) - place_order_time

                # Compute the fees in quoteasset and BNB
                quote_fee = Decimal(buy_order_result['cummulativeQuoteQty'])*Decimal(0.075)/Decimal(100)
                BNB_fee   = quote_fee / Decimal(exchange.GetLastestPriceOfPair(pair='BNB'+bot['quoteasset']))

                database.SaveOrder(quoteasset        = bot['quoteasset'],
                                   order_result      = buy_order_result,
                                   hold_duration     = '-',
                                   profit            = '-',
                                   quote_fee         = format(round(quote_fee, bot['quoteAssetPrecision']), 'f'),
                                   BNB_fee           = format(round(BNB_fee,   bot['BNB_precision']),       'f'),
                                   profit_minus_fees = '-',
                                   time_to_fill      = str(timedelta_to_fill))

                database.UpdateBot(pair             = pair,
                                   status           = 'Looking to exit',
                                   quoteBalance     = buy_order_result['cummulativeQuoteQty'],
                                   baseBalance      = buy_order_result['executedQty'],
                                   last_order_date  = datetime.utcfromtimestamp(int(buy_order_result['transactTime'])/1000).strftime("%Y-%m-%d %H:%M:%S"),
                                   number_of_orders = +1,                                                                # Added
                                   bot_quote_fees   = format(round(quote_fee, bot['quoteAssetPrecision']), 'f'),         # Added to the current bot_quote_fees
                                   bot_BNB_fees     = format(round(BNB_fee,   bot['BNB_precision']),       'f'))         # Added


                # Update the balances count
                database.UpdateAccountBalance(quoteasset                 = bot['quoteasset'],
                                              real_quote_balance         = exchange.GetAccountBalance(quoteasset=bot['quoteasset']),
                                              real_profit                = format(round(Decimal('0'), bot['quoteAssetPrecision']), 'f'),
                                              internal_quote_balance     = format(round(-Decimal(buy_order_result['cummulativeQuoteQty']), bot['quoteAssetPrecision']), 'f'),           # Added
                                              internal_profit            = format(round(Decimal('0'), bot['quoteAssetPrecision']), 'f'),                                                    # Added
                                              internal_quote_fees        = format(round(quote_fee,    bot['quoteAssetPrecision']), 'f'),                                                    # Added
                                              internal_BNB_fees          = format(round(BNB_fee,      bot['BNB_precision']),       'f'),                                                    # Added
                                              internal_profit_minus_fees = format(round(Decimal('0'), bot['quoteAssetPrecision']), 'f'),                                                    # Added
                                              quoteAssetPrecision        = bot['quoteAssetPrecision'],
                                              BNB_Precision              = bot['BNB_precision'])



                # Send a text to telegram
                # self.send_text_to_telegram(text)

                dict_to_fill[pair] = [datetime.utcfromtimestamp(int(buy_order_result['transactTime'])/1000).strftime("%H:%M:%S"),
                                      Decimal(buy_order_result['price']).normalize(),
                                      Decimal(buy_order_result['executedQty']).normalize(),
                                      Decimal(buy_order_result['cummulativeQuoteQty']).normalize()]

                return dict_to_fill

        else:
            database.UpdateBot(pair=pair, status='', quoteBalance='')


    def sell_order(self, bot:dict, dict_to_fill:dict):
        """ The bot successfully placed a buy order and is now look for a sell signal.
            If it finds one, places a sell order. """

        pair 	    = bot['pair']
        sp          = yaspin()
        sp.color    = 'red'
        sp.text     = "\tChecking for sell signals on " + pair + "."
        df    	    = exchange.GetPairKlines(pair, bot['timeframe'], candles=200)

        if df.empty:
            return None

        # signal = strategy.find_signal(df=df)		# check for a signal (returns 'buy'/'sell')
        signal = random.choice(['buy', 'sell'])

        if signal == 'sell':

            # Mandatory parameters to send to the endpoint:
            sell_order_parameters = dict(symbol = pair,
                                         side   = "SELL",
                                         type   = "LIMIT")

            quoteOrderQty, sell_price, quantity = Decimal('0'), Decimal('0'), Decimal('0')
            market_price  = exchange.GetLastestPriceOfPair(pair=pair)                                                       # market_price is a string

            # Additional mandatory parameters based on type
            if sell_order_parameters['type'] == 'MARKET':
                quoteOrderQty = Decimal(database.GetBot(pair)['quoteBalance'])
                sell_price    = market_price
                # Additional mandatory parameter : quoteOrderQty
                sell_order_parameters['quoteOrderQty'] = format(round(quoteOrderQty, bot['quoteAssetPrecision']), 'f')			# specifies the amount the user wants to spend (when buying) or receive (when selling) of the quote asset; the correct quantity will be determined based on the market liquidity and quoteOrderQty

            elif sell_order_parameters['type'] == 'LIMIT':
                sell_price    = exchange.RoundToValidPrice(pair=pair, price=Decimal(market_price)*Decimal(1.01))                # buy_price is a Decimal
                quantity      = exchange.RoundToValidQuantity(pair=pair, quantity=Decimal(bot['quoteBalance'])/sell_price)      # quantity  is a Decimal
                quoteOrderQty = sell_price*quantity

                # Additional mandatory parameters : price & quantity
                sell_order_parameters['timeInForce'] = 'GTC'		 		                                                     # 'GTC' (Good-Till-Canceled), 'IOC' (Immediate-or-Cancel) (part or all of the order) or 'FOK' (Fill-or-Kill) (whole order)
                sell_order_parameters['price'] 	     = format(round(sell_price, bot['baseAssetPrecision']), 'f')
                sell_order_parameters['quantity']    = format(round(quantity,   bot['baseAssetPrecision']), 'f')

            # Simulate a sell order
            place_order_time  = datetime.utcnow()
            sell_order_result = exchange.PlaceOrder(order_params=sell_order_parameters, test_order=self.paper_trading)

            if self.paper_trading:
                dummy_sell_order_result = {"symbol"               : pair,
                                           "orderId"              : str(uuid1()),
                                           "orderListId"          : -1,
                                           "clientOrderId"        : "Sell_6gCrw2kRUAF9CvJDGP16IP",
                                           "transactTime"         : str(exchange.GetServerTime()),
                                           "price"                : format(round(sell_price,    bot['quotePrecision']),      'f'),
                                           "origQty"              : format(round(quantity,      bot['baseAssetPrecision']),  'f'),
                                           "executedQty"          : format(round(quantity,      bot['baseAssetPrecision']),  'f'),
                                           "cummulativeQuoteQty"  : format(round(quoteOrderQty, bot['quoteAssetPrecision']), 'f'),
                                           "status"               : "FILLED",
                                           "timeInForce"          : "GTC",
                                           "type"                 : sell_order_parameters['type'],
                                           "side"                 : sell_order_parameters['side']}
                sell_order_result = {**dummy_sell_order_result, **sell_order_result}

            if "code" in sell_order_result:
                formatted_transactTime = datetime.utcfromtimestamp(int(sell_order_result['transactTime'])/1000).strftime('%H:%M:%S')
                print(f"\t{formatted_transactTime} - Error in placing a sell {'test' if self.paper_trading else ''} order on {pair} at {sell_price} :/")
                print(pair, sell_order_result)
                return None

            else:
                # Compute the time taken to fill the order
                timedelta_to_fill = datetime.utcfromtimestamp(int(sell_order_result['transactTime'])/1000) - place_order_time

                # Compute the profit from the trade, in quoteasset
                profit            = Decimal(sell_order_result['cummulativeQuoteQty']) - Decimal(database.GetBot(pair)['quoteBalance'])                     # On a pas encore update la balance donc on sait pour combien de quoteasset a acheté
                quote_fee         = Decimal(sell_order_result['cummulativeQuoteQty'])*Decimal(0.075)/Decimal(100)
                profit_minus_fees = profit - quote_fee - Decimal(dict(list(database.GetOrdersOfBot(pair))[-1])['quote_fee'])
                BNB_fee           = quote_fee / Decimal(exchange.GetLastestPriceOfPair(pair='BNB'+bot['quoteasset']))

                # How long we have been holding the asset for
                hold_timedelta = datetime.utcfromtimestamp(int(sell_order_result['transactTime'])/1000) - datetime.strptime(dict(database.GetBot(pair=pair))['last_order_date'], "%Y-%m-%d %H:%M:%S")

                text = "\t{time} - Success in placing a sell {test} order on {pair} \t : sold {quantity} {base} at {price} {quoteasset} for {quoteQty} {quoteasset}. \t Profit : {profit} {quoteasset}".format(time       = datetime.utcfromtimestamp(int(sell_order_result['transactTime'])/1000).strftime("%H:%M:%S"),
                                                                                                                                                                                                               test       = 'test' if self.paper_trading else '',
                                                                                                                                                                                                               pair       = pair,
                                                                                                                                                                                                               quantity   = sell_order_result['executedQty'],
                                                                                                                                                                                                               base       = pair.replace(bot['quoteasset'], ''),
                                                                                                                                                                                                               price      = sell_price,
                                                                                                                                                                                                               quoteasset = bot['quoteasset'],
                                                                                                                                                                                                               quoteQty   = sell_order_result['cummulativeQuoteQty'],
                                                                                                                                                                                                               profit     = profit)
                print(text)

                database.SaveOrder(quoteasset        = bot['quoteasset'],
                                   order_result      = sell_order_result,
                                   hold_duration     = str(hold_timedelta),
                                   profit            = format(round(profit,            bot['quoteAssetPrecision']), 'f'),
                                   quote_fee         = format(round(quote_fee,         bot['quoteAssetPrecision']), 'f'),
                                   BNB_fee           = format(round(BNB_fee,           bot['BNB_precision']),       'f'),
                                   profit_minus_fees = format(round(profit_minus_fees, bot['quoteAssetPrecision']), 'f'),
                                   time_to_fill      = str(timedelta_to_fill))

                database.UpdateBot(pair                  = pair,
                                   status                = 'Looking to enter',
                                   quoteBalance          = '',
                                   baseBalance           = '',
                                   last_order_date       = datetime.utcfromtimestamp(int(sell_order_result['transactTime'])/1000).strftime("%Y-%m-%d %H:%M:%S"),
                                   last_profit           = str(profit),
                                   bot_profit            = format(round(profit,            bot['quoteAssetPrecision']), 'f'),       # Added to the current bot_profit
                                   bot_quote_fees        = format(round(quote_fee,         bot['quoteAssetPrecision']), 'f'),       # Added
                                   bot_BNB_fees          = format(round(BNB_fee,           bot['BNB_precision']),       'f'),       # Added
                                   bot_profit_minus_fees = format(round(profit_minus_fees, bot['quoteAssetPrecision']), 'f'),       # Added
                                   number_of_orders      = +1)
                if profit>0:
                    database.UpdateBot(pair=pair, profitable_sells=+1)
                else:
                    database.UpdateBot(pair=pair, unprofitable_sells=+1)

                # Update the internal balances count
                database.UpdateAccountBalance(quoteasset                 = bot['quoteasset'],
                                              real_quote_balance         = exchange.GetAccountBalance(quoteasset=bot['quoteasset']),
                                              real_profit                = format(round(Decimal(exchange.GetAccountBalance(bot['quoteasset']))/Decimal(database.GetStartBalance(quoteasset=bot['quoteasset']))*100, 8), 'f'),
                                              internal_quote_balance     = format(round(Decimal(sell_order_result['cummulativeQuoteQty']), bot['quoteAssetPrecision']), 'f'),       # Added
                                              internal_profit            = format(round(profit,    bot['quoteAssetPrecision']), 'f'),                                                   # Added
                                              internal_quote_fees        = format(round(quote_fee, bot['quoteAssetPrecision']), 'f'),                                                   # Added
                                              internal_BNB_fees          = format(round(BNB_fee,   bot['BNB_precision']),       'f'),                                                   # Added
                                              internal_profit_minus_fees = format(round(profit_minus_fees, bot['quoteAssetPrecision']), 'f'),                                           # Added
                                              quoteAssetPrecision        = bot['quoteAssetPrecision'],
                                              BNB_Precision              = bot['BNB_precision'])

                # Send a text to telegram
                # self.send_text_to_telegram(text)

                dict_to_fill[pair] = [datetime.utcfromtimestamp(int(sell_order_result['transactTime'])/1000).strftime("%H:%M:%S"),
                                      Decimal(sell_order_result['price']).normalize(),
                                      Decimal(sell_order_result['executedQty']).normalize(),
                                      Decimal(sell_order_result['cummulativeQuoteQty']).normalize()]

                return dict_to_fill


    def send_text_to_telegram(self, bot_message):

        send_text  = f'https://api.telegram.org/bot{self.bot_token}/sendMessage?chat_id={self.bot_chatID}&parse_mode=Markdown&text={bot_message}'
        response = requests.get(send_text)
        return response.json()


    def set_pairs_to_trade_on(self)->bool:
        """ Gets the name of the pairs that we will be trading on
            and set the corresponding bots as active, if enough money in account. """

        sp       = yaspin()
        sp.color = 'green'

        trading_pairs = dict()
        sorted_pairs  = dict()
        # holding     = dict()

        open_positions = self.check_open_positions()     # {'ETH':[], 'BTC':[]}                                        # Check for open positions (bots that are holding a coin)

        account_balance = dict()

        for quoteasset in self.existing_quoteassets:
            sp.text = "Processing " + quoteasset + " pairs..."
            sp.start()

            account_balance[quoteasset] = Decimal(database.GetAccountBalance(quoteasset=quoteasset, real_or_internal='internal'))

            # If we have room for more bots
            if len(open_positions[quoteasset]) < BotsPerQuoteAsset:
                # Find the best X pairs by volume
                sorted_pairs[quoteasset] = self.sort_by_daily_volume(quoteasset=quoteasset)                                                                                      # best_pairs = {'ETH' : ['HOTETH', 'DENTETH', 'NPXSETH', 'NCASHETH', 'KEYETH', 'ZILETH', 'TRXETH', 'SCETH', 'MFTETH', 'VETETH'],
                # Set the pairs to trade on for each quoteasset : the holding pairs, completed with the best pairs per volume if needed.                                    #               'BTC' : ['HOTBTC', 'VETBTC', 'ZILBTC', 'MBLBTC', 'MATICBTC', 'FTMBTC', 'TRXBTC', 'IOSTBTC', 'DOGEBTC', 'SCBTC']}
                # Do not trade on the quoteassets and BNB, to avoid balances problems.
                filtered = (k for k in sorted_pairs[quoteasset] if k not in open_positions[quoteasset] if not k.startswith(tuple([q for q in self.existing_quoteassets])) if not k.startswith('BNB'))
                trading_pairs[quoteasset] = open_positions[quoteasset] + list(islice(filtered, BotsPerQuoteAsset-len(open_positions[quoteasset])))

                min_amount = sum(Decimal(exchange.GetMinNotional(pair)) for pair in trading_pairs[quoteasset] if pair not in open_positions[quoteasset])
                # holding[quoteasset] = sum(Decimal(dict(bot)['quoteBalance']) for bot in database.GetAllBots() if dict(bot)['quoteasset']==quoteasset and dict(bot)['quoteBalance'] != '')           # How much bots are holding atm on this quoteasset.

                sp.stop()

                # Check if balance is enough to trade on the completing pairs
                if account_balance[quoteasset] > min_amount:
                    allocation = account_balance[quoteasset] / (len(trading_pairs[quoteasset])-len(open_positions[quoteasset]))
                    print(f"Paper trading on the top {BotsPerQuoteAsset} pairs by volume on {quoteasset} : {trading_pairs[quoteasset]}")
                    if open_positions[quoteasset]:
                        print(f"(using {open_positions[quoteasset]} which are still trying to sell.)")
                    print("{quoteasset} balance : {quoteasset_balance} {quoteasset}. Each new bot is given {balance} {quoteasset} to trade and has been set as active.".format(quoteasset_balance = account_balance[quoteasset].normalize(),
                                                                                                                                                                               balance            = format(round(allocation, 8), 'f'),
                                                                                                                                                                               quoteasset         = quoteasset))
                    # Set the bots as active
                    for pair in trading_pairs[quoteasset]:
                        if pair not in open_positions[quoteasset]:
                            database.UpdateBot(pair         = pair,
                                               status       = 'Looking to enter',
                                               quoteBalance = format(round(allocation, dict(database.GetBot(pair))['quoteAssetPrecision']), 'f'))

                else:
                    print("{quoteasset} balance : {balance} {quoteasset}. Not enough to trade on {nbpairs} pairs. Minimum amount (not including fees) required : {min_amount}{quoteasset}.".format(balance    = account_balance[quoteasset].normalize(),
                                                                                                                                                                                                   nbpairs    = len(trading_pairs[quoteasset])-len(open_positions[quoteasset]),
                                                                                                                                                                                                   quoteasset = quoteasset,
                                                                                                                                                                                                   min_amount = min_amount.normalize()))
            else:
                # If we already have the max number of bots, do nothing
                sp.stop()
                print(f"No room left for new bots on {quoteasset}. Trading on {open_positions[quoteasset]}")

        if not [bot["status"] for bot in database.GetAllBots()]:
            print("\nYou can't trade on any quoteAsset. Please change the starting balance and try again ! \n_________________________________________________________________________________________________________________________")
            return False


    def wait_for_candle(self)->bool:
        """ Pauses the program until the next candle arrives.
            Just before the candle arrives, sets the pairs to trade on. """

        sp = yaspin()
        sp.start()

        parsed_timeframe = re.findall(r'[A-Za-z]+|\d+', timeframe)      # Separates '30m' in ['30', 'm']
        next_candle = None
        now = datetime.now()

        # Find the time of the next candle
        if parsed_timeframe[1] == 'm':
            if now.minute < 60-int(parsed_timeframe[0]):
                min_till_candle = int(parsed_timeframe[0])-(now.minute % int(parsed_timeframe[0]))
                next_candle     = datetime(now.year, now.month, now.day, now.hour, now.minute + min_till_candle, 0)
                sp.text = "Waiting for the next " + timeframe + " candle (at " + next_candle.strftime("%H:%M") + ") to start trading."
            else:
                next_candle = datetime(now.year, now.month, now.day, now.hour+1, 0, 0)
                sp.text = "Waiting for the next " + timeframe + " candle (at " + next_candle.strftime("%H:%M") + ") to start trading."

        elif parsed_timeframe[1] == 'h':
            next_candle = datetime(now.year, now.month, now.day, now.hour+int(parsed_timeframe[0]), 0, 0)
            sp.text = "Waiting for the next " + timeframe + " candle (at " + next_candle.strftime("%H:%M") + ") to start trading."

        # Update the pairs just before the candle arrives
        if now < next_candle-timedelta(seconds=50):
            pause.until(next_candle-timedelta(seconds=50))
        else:
            sp.text = "Arrived too late for the first candle. Waiting for the next one."
            pause.until(next_candle)        # Cas de la toute premiere candle : on ne fait rien si on arrive moins de 50 secondes avant, on attend la candle suivante.

        sp.stop()
        sp.hide()

        print("_______________________________________")
        print(colorama.Fore.GREEN + "Candle : " + str(next_candle.strftime("%Y-%m-%d %H:%M")) + " (local time).")
        self.set_pairs_to_trade_on()          # Takes ~40secs to run

        sp.start()
        sp.show()
        sp.text = "Candle is coming..."
        # If time left, pause until the candle arrives
        pause.until(next_candle-timedelta(seconds=5))
        sp.stop()
        sp.hide()

        # Display counter for the last 5 secs
        for remaining in range(5, 0, -1):
            sys.stdout.write("\r")
            sys.stdout.write("Candle is coming in {:2d} seconds...".format(remaining))
            sys.stdout.flush()
            time.sleep(1)
        # print('\r')
        sys.stdout.write("\r")          # Erase the last line

        return False

    @staticmethod
    def check_open_positions()->dict:

        open_positions = dict()
        # Create a list of all the quoteassets present in the database
        quoteassets = list(set([dict(bot)['quoteasset'] for bot in database.GetAllBots()]))

        for quoteasset in quoteassets:
            open_positions[quoteasset] = [dict(bot)['pair'] for bot in database.GetAllBots() if dict(bot)['status']=='Looking to exit' and dict(bot)['quoteasset']==quoteasset]                                 # {'ETH':[], 'BTC':[]}

        return open_positions           # {'ETH':[], 'BTC':[]}

    @staticmethod
    def sort_by_daily_volume(quoteasset:str)->list:
        """ Returns a dict of the best X pairs per quoteasset based on their rolling-24H volume of quoteasset.
            Therefore, we can access the top trading pairs in terms of quoteasset volume and set our bots on them. """

        pairs = exchange.GetNameOfPairs_WithQuoteasset(quoteAssets=[quoteasset])		# {'BTC':['ETHBTC', 'LTCBTC',...]}
        if pairs == {}:
            sys.exit("Could not get the pairs from Binance to sort them by day volume. Exiting the script.")

        stats = dict()

        def GetVolume(pair:str):
            pair_info = exchange.GetMetadataOfPair(pair)

            if pair_info == {}:
                print(f"Could not get {pair} info from Binance to get its volume.")

            elif pair_info and pair_info['status'] == 'TRADING':
                stats[pair] = Decimal(exchange.Get24hrTicker(pair)['quoteVolume'])

        pool  = Pool()
        func1 = partial(GetVolume)
        pool.map(func1, pairs[quoteasset])
        pool.close()
        pool.join()

        sorted_volumes  = {k: v for k, v in sorted(stats.items(), key=lambda item: item[1], reverse=True)}
        sorted_pairs_list = list(sorted_volumes.keys())

        return sorted_pairs_list              # ['NPXSETH', 'DENTETH', 'HOTETH', 'KEYETH', 'NCASHETH', 'MFTETH', 'TRXETH', 'SCETH', 'ZILETH', 'STORMETH', ...]

    @staticmethod
    def print_trades_sequence(buys:dict, sells:dict):
        """ Prints the results of the last search for signals. """

        print("\n\t({nb_buys} Buys, {nb_sells} Sells.)".format(nb_buys=len(buys), nb_sells=len(sells)))
        print("")



if __name__ == "__main__":



    # Parameters
    timeframe         = '1m'
    WantedQuoteAssets = ['BTC', 'ETH']
    BotsPerQuoteAsset = 10

    # # Delete the bots on a specific quoteAsset.
    # database.DeleteBots(quoteasset='')

    # Start trading
    Trading(paper_trading=True).main_loop()
