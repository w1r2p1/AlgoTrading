from Strategies import *
import Settings

from uuid     import uuid1
from decimal  import Decimal, getcontext
from yaspin   import yaspin
from multiprocessing.pool import ThreadPool as Pool
from functools import partial
from datetime  import datetime, timedelta
import re
import random
from itertools import islice
import sys
from colorama import init
from colorama import Fore
import configparser as cfg
import requests
import pause
import time

def read_token_from_config_file(config):
    parser = cfg.ConfigParser()
    parser.read(config)
    return parser.get('creds', 'token')


class PaperTrading:

    def __init__(self, exchange, database):
        self.exchange   = exchange
        self.database   = database
        self.skips      = 0
        self.bot_token  = read_token_from_config_file("config.cfg")
        self.bot_chatID = '456212622'


    def StartExecution(self):

        database = self.database
        exchange = self.exchange

        self.SetAccountBalances()

        init()
        print("\n_______________________________________________________________________________________________")
        print(Fore.GREEN + "PAPER TRADING IS LIVE.")
        print("Timeframe : " + Settings.parameters['timeframe'] + "\n")

        while True:

            # Pause the program until the next candle arrives. Just before the candle arrives, set the pairs to trade on.
            wait = self.WaitForCandle()

            if not wait:
                # When the candle appears, start scanning
                print("\nTrading with a clocks difference of {time} ms.".format(time=exchange.TimeDifferenceWithBinance()))

                buys  = dict()
                pool  = Pool()
                func1 = partial(self.EntryOrder, dict_to_fill=buys)
                pool.map(func1, [dict(bot) for bot in database.GetAllBots() if dict(bot)['status']=='Looking to enter'])
                pool.close()
                pool.join()
                print("\tChecked all pairs for buy  signals.")

                sells = dict()
                pool  = Pool()
                func2 = partial(self.ExitOrder, dict_to_fill=sells)
                pool.map(func2, [dict(bot) for bot in database.GetAllBots() if dict(bot)['status']=='Looking to exit'])
                pool.close()
                pool.join()
                print("\tChecked all pairs for sell signals.")

                # Summary of the search
                self.PrintTradeSequence(buys=buys, sells=sells)
                # break


    def EntryOrder(self, bot:dict, dict_to_fill:dict):
        """ The bot looks for a buy signal. If yes, places an order. """

        exchange = self.exchange
        database = self.database

        pair 	   = bot['pair']
        sp         = yaspin()
        sp.color   = 'red'
        sp.text    = "\tChecking for buy signals on " + pair + ", timeframe " + bot['timeframe'] + "."
        df         = exchange.GetPairKlines(pair=pair, timeframe=bot['timeframe'], candles=200)				                            # get dataframe

        if df.empty:
            return None

        buy_signal = strategies_dict[bot['strategy']](df=df, i=len(df['close'])-1, signal='buy', fast_period=50, slow_period=200)		# check for a buy signal (True/False)
        buy_signal = random.choice([True, False])

        if buy_signal:

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
                buy_order_parameters['quantity']    = format(round(quantity,   bot['baseAssetPrecision']), 'f')

            # Simumlate a buy from exchange
            place_order_time = datetime.utcnow()
            buy_order_result = exchange.PlaceOrder(order_params=buy_order_parameters, test_order=bot['test_orders'])

            # Dummy buy order result
            dummyBuyorder_result = {"symbol"               : pair,
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


            if "code" in buy_order_result:
                print("\t{time} - Error in placing a buy test order on {pair} at {buy_price} :/".format(time      = datetime.utcfromtimestamp(Decimal(dummyBuyorder_result['transactTime'])/1000).strftime("%H:%M:%S"),
                                                                                                        pair      = pair,
                                                                                                        buy_price = buy_price))
                database.UpdateBot(pair=pair, status='', quoteBalance='')
                print(pair, buy_order_result)
                return None

            else:
                text = "\t{time} - Success in placing a buy test order on {pair} \t: bought {quantity} {base} \tat {price} {quoteasset} for {quoteQty} {quoteasset}.".format(time       = datetime.utcfromtimestamp(Decimal(dummyBuyorder_result['transactTime'])/1000).strftime("%H:%M:%S"),
                                                                                                                                                                             pair       = pair,
                                                                                                                                                                             quantity   = dummyBuyorder_result['executedQty'],
                                                                                                                                                                             base       = pair.replace(bot['quoteasset'], ''),
                                                                                                                                                                             price      = buy_price,
                                                                                                                                                                             quoteasset = bot['quoteasset'],
                                                                                                                                                                             quoteQty   = dummyBuyorder_result['cummulativeQuoteQty'])
                print(text)

                # Compute the time taken to fill the order
                timedelta_to_fill = datetime.utcfromtimestamp(int(dummyBuyorder_result['transactTime'])/1000) - place_order_time

                # Compute the fees in quoteasset and BNB
                quote_fee = Decimal(dummyBuyorder_result['cummulativeQuoteQty'])*Decimal(0.075)/Decimal(100)
                BNB_fee   = quote_fee / Decimal(exchange.GetLastestPriceOfPair(pair='BNB'+bot['quoteasset']))

                database.SaveOrder(quoteasset        = bot['quoteasset'],
                                   order_result      = dummyBuyorder_result,
                                   hold_duration     = '-',
                                   profit            = '-',
                                   quote_fee         = format(round(quote_fee, bot['quoteAssetPrecision']), 'f'),
                                   BNB_fee           = format(round(BNB_fee,   bot['BNB_precision']),       'f'),
                                   profit_minus_fees = '-',
                                   time_to_fill      = str(timedelta_to_fill))

                database.UpdateBot(pair             = pair,
                                   status           = 'Looking to exit',
                                   quoteBalance     = dummyBuyorder_result['cummulativeQuoteQty'],
                                   baseBalance      = dummyBuyorder_result['executedQty'],
                                   last_order_date  = datetime.utcfromtimestamp(Decimal(dummyBuyorder_result['transactTime'])/1000).strftime("%Y-%m-%d %H:%M:%S"),
                                   number_of_orders = +1,                                                                # Added
                                   bot_quote_fees   = format(round(quote_fee, bot['quoteAssetPrecision']), 'f'),         # Added to the current bot_quote_fees
                                   bot_BNB_fees     = format(round(BNB_fee,   bot['BNB_precision']),       'f'))         # Added


                # Update the balances count
                database.UpdateAccountBalance(quoteasset                 = bot['quoteasset'],
                                              real_quote_balance         = exchange.GetAccountBalance(quoteasset=bot['quoteasset']),
                                              real_profit                = format(round(Decimal('0'), bot['quoteAssetPrecision']), 'f'),
                                              internal_quote_balance     = format(round(-Decimal(dummyBuyorder_result['cummulativeQuoteQty']), bot['quoteAssetPrecision']), 'f'),           # Added
                                              internal_profit            = format(round(Decimal('0'), bot['quoteAssetPrecision']), 'f'),                                                    # Added
                                              internal_quote_fees        = format(round(quote_fee,    bot['quoteAssetPrecision']), 'f'),                                                    # Added
                                              internal_BNB_fees          = format(round(BNB_fee,      bot['BNB_precision']),       'f'),                                                    # Added
                                              internal_profit_minus_fees = format(round(Decimal('0'), bot['quoteAssetPrecision']), 'f'),                                                    # Added
                                              quoteAssetPrecision        = bot['quoteAssetPrecision'],
                                              BNB_Precision              = bot['BNB_precision'])



                # Send a text to telegram
                # self.telegram_bot_sendtext(text)

                dict_to_fill[pair] = [datetime.utcfromtimestamp(Decimal(dummyBuyorder_result['transactTime'])/1000).strftime("%H:%M:%S"),
                                      Decimal(dummyBuyorder_result['price']).normalize(),
                                      Decimal(dummyBuyorder_result['executedQty']).normalize(),
                                      Decimal(dummyBuyorder_result['cummulativeQuoteQty']).normalize()]

                return dict_to_fill

        else:
            database.UpdateBot(pair=pair, status='', quoteBalance='')


    def ExitOrder(self, bot:dict, dict_to_fill:dict):
        """ The bot successfully placed a buy order and is now look for a sell signal.
            If it finds one, places a sell order. """

        exchange = self.exchange
        database = self.database

        pair 	    = bot['pair']
        sp          = yaspin()
        sp.color    = 'red'
        sp.text     = "\tChecking for sell signals on " + pair + "."
        df    	    = exchange.GetPairKlines(pair, bot['timeframe'], candles=200)				                                        # get dataframe

        if df.empty:
            return None

        sell_signal = strategies_dict[bot['strategy']](df=df, i=len(df['close'])-1, signal='sell', fast_period=50, slow_period=200)		# check for a sell signal
        sell_signal = random.choice([True, False])

        database.UpdateBot(pair=pair, current_status='Looking to exit')

        if sell_signal is not False:

            # Mandatory parameters to send to the endpoint:
            sell_order_parameters = dict(symbol = pair,
                                         side   = "SELL",
                                         type   = "LIMIT")

            quoteOrderQty, sell_price, quantity = Decimal('0'), Decimal('0'), Decimal('0')

            # Additional mandatory parameters based on type
            if sell_order_parameters['type'] == 'MARKET':
                quoteOrderQty = Decimal(database.GetBot(pair)['quoteBalance'])

                # Additional mandatory parameter : quoteOrderQty
                sell_order_parameters['quoteOrderQty'] = format(round(quoteOrderQty, bot['quoteAssetPrecision']), 'f')			# specifies the amount the user wants to spend (when buying) or receive (when selling) of the quote asset; the correct quantity will be determined based on the market liquidity and quoteOrderQty

            elif sell_order_parameters['type'] == 'LIMIT':
                market_price  = exchange.GetLastestPriceOfPair(pair=pair)                                                       # market_price is a string
                sell_price    = exchange.RoundToValidPrice(pair=pair, price=Decimal(market_price)*Decimal(1.01))                # buy_price is a Decimal
                quantity      = exchange.RoundToValidQuantity(pair=pair, quantity=Decimal(bot['quoteBalance'])/sell_price)      # quantity  is a Decimal
                quoteOrderQty = sell_price*quantity

                # Additional mandatory parameters : price & quantity
                sell_order_parameters['timeInForce'] = 'GTC'		 		                                                     # 'GTC' (Good-Till-Canceled), 'IOC' (Immediate-or-Cancel) (part or all of the order) or 'FOK' (Fill-or-Kill) (whole order)
                sell_order_parameters['price'] 	     = format(round(sell_price, bot['baseAssetPrecision']), 'f')
                sell_order_parameters['quantity']    = format(round(quantity,   bot['baseAssetPrecision']), 'f')


            # Simulate a sell order
            place_order_time  = datetime.utcnow()
            sell_order_result = exchange.PlaceOrder(order_params=sell_order_parameters, test_order=bot['test_orders'])

            # Dummy sell order result
            dummySellorder_result = {"symbol"               : pair,
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

            if "code" in sell_order_result:
                print("\t{time} - Error in placing a sell test order on {pair} at {sell_price} !".format(time       = datetime.utcfromtimestamp(Decimal(dummySellorder_result['transactTime'])/1000).strftime("%H:%M:%S"),
                                                                                                         pair       = pair,
                                                                                                         sell_price = sell_price))
                print(pair, sell_order_result)
                return None

            else:
                # Compute the time taken to fill the order
                timedelta_to_fill = datetime.utcfromtimestamp(int(dummySellorder_result['transactTime'])/1000) - place_order_time

                # Compute the profit from the trade, in quoteasset
                profit            = Decimal(dummySellorder_result['cummulativeQuoteQty']) - Decimal(database.GetBot(pair)['quoteBalance'])                     # On a pas encore update la balance donc on sait pour combien de quoteasset a acheté
                quote_fee         = Decimal(dummySellorder_result['cummulativeQuoteQty'])*Decimal(0.075)/Decimal(100)
                profit_minus_fees = profit - quote_fee - Decimal(dict(list(database.GetOrdersOfBot(pair))[-1])['quote_fee'])
                BNB_fee           = quote_fee / Decimal(exchange.GetLastestPriceOfPair(pair='BNB'+bot['quoteasset']))

                # How long we have been holding the asset for
                hold_timedelta = datetime.utcfromtimestamp(Decimal(dummySellorder_result['transactTime'])/1000) - datetime.strptime(dict(database.GetBot(pair=pair))['last_order_date'], "%Y-%m-%d %H:%M:%S")

                text = "\t{time} - Success in placing a sell test order on {pair} \t : sold {quantity} {base} at {price} {quoteasset} for {quoteQty} {quoteasset}. \t Profit : {profit} {quoteasset}".format(time       = datetime.utcfromtimestamp(Decimal(dummySellorder_result['transactTime'])/1000).strftime("%H:%M:%S"),
                                                                                                                                                                                                             pair       = pair,
                                                                                                                                                                                                             quantity   = dummySellorder_result['executedQty'],
                                                                                                                                                                                                             base       = pair.replace(bot['quoteasset'], ''),
                                                                                                                                                                                                             price      = sell_price,
                                                                                                                                                                                                             quoteasset = bot['quoteasset'],
                                                                                                                                                                                                             quoteQty   = dummySellorder_result['cummulativeQuoteQty'],
                                                                                                                                                                                                             profit     = profit)
                print(text)

                database.SaveOrder(quoteasset        = bot['quoteasset'],
                                   order_result      = dummySellorder_result,
                                   hold_duration     = str(hold_timedelta),
                                   profit            = format(round(profit,            bot['quoteAssetPrecision']), 'f'),
                                   quote_fee         = format(round(quote_fee,         bot['quoteAssetPrecision']), 'f'),
                                   BNB_fee           = format(round(BNB_fee,           bot['BNB_precision']),       'f'),
                                   profit_minus_fees = format(round(profit_minus_fees, bot['quoteAssetPrecision']), 'f'),
                                   time_to_fill      = str(timedelta_to_fill))

                database.UpdateBot(pair                  = pair,
                                   status                = '',
                                   quoteBalance          = '',
                                   baseBalance           = '',
                                   last_order_date       = datetime.utcfromtimestamp(Decimal(dummySellorder_result['transactTime'])/1000).strftime("%Y-%m-%d %H:%M:%S"),
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
                                              internal_quote_balance     = format(round(Decimal(dummySellorder_result['cummulativeQuoteQty']), bot['quoteAssetPrecision']), 'f'),       # Added
                                              internal_profit            = format(round(profit,    bot['quoteAssetPrecision']), 'f'),                                                   # Added
                                              internal_quote_fees        = format(round(quote_fee, bot['quoteAssetPrecision']), 'f'),                                                   # Added
                                              internal_BNB_fees          = format(round(BNB_fee,   bot['BNB_precision']),       'f'),                                                   # Added
                                              internal_profit_minus_fees = format(round(profit_minus_fees, bot['quoteAssetPrecision']), 'f'),                                           # Added
                                              quoteAssetPrecision        = bot['quoteAssetPrecision'],
                                              BNB_Precision              = bot['BNB_precision'])

                # Send a text to telegram
                # self.telegram_bot_sendtext(text)

                dict_to_fill[pair] = [datetime.utcfromtimestamp(Decimal(dummySellorder_result['transactTime'])/1000).strftime("%H:%M:%S"),
                                      Decimal(dummySellorder_result['price']).normalize(),
                                      Decimal(dummySellorder_result['executedQty']).normalize(),
                                      Decimal(dummySellorder_result['cummulativeQuoteQty']).normalize()]

                return dict_to_fill


    def telegram_bot_sendtext(self, bot_message):

        bot_token  = self.bot_token
        bot_chatID = self.bot_chatID
        send_text  = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message

        response = requests.get(send_text)

        return response.json()


    def SetAccountBalances(self):
        """ If we never traded on a quoteasset, set the start of it's internal balance. """

        database = self.database
        exchange = self.exchange

        # List of all the quoteassets present in the database
        ExistingQuoteassets = list(set([dict(bot)['quoteasset'] for bot in database.GetAllBots()]))             # ['ETH', 'BTC']

        # Creates an account balance for the quoteassets if they are not already existing
        for quoteasset in ExistingQuoteassets:
            database.InitiateAccountBalance(quoteasset             = quoteasset,
                                            started_with           = format(round(Settings.parameters['start_balance_if_test_orders'], 8), 'f'),
                                            real_quote_balance     = exchange.GetAccountBalance(quoteasset),
                                            internal_quote_balance = format(round(Settings.parameters['start_balance_if_test_orders'], 8), 'f'))


    def CheckOpenPositions(self)->dict:

        database = self.database

        open_positions = dict()
        # Create a list of all the quoteassets present in the database
        quoteassets = list(set([dict(bot)['quoteasset'] for bot in database.GetAllBots()]))

        for quoteasset in quoteassets:
            open_positions[quoteasset] = [dict(bot)['pair'] for bot in database.GetAllBots() if dict(bot)['status']=='Looking to exit' and dict(bot)['quoteasset']==quoteasset]                                 # {'ETH':[], 'BTC':[]}
            # open_positions[quoteasset] = [{dict(bot)['pair']:dict(bot)['balance']} for bot in database.GetAllBots() if dict(bot)['status']=='Looking to exit' and dict(bot)['quoteasset']==quoteasset]        # {'ETH':[{}, {}], 'BTC':[{}, {}]}

        return open_positions           # {'ETH':[], 'BTC':[]}


    def Skips(self):
        # If we could not get the info of more than 5 pairs, assume the connexion is down and exit the script.
        self.skips += 1
        if self.skips > 5:
            # return None
            sys.exit("Could not get the info from Binance on more than 5 pairs, to get their volume.")


    def SortByDayVolume(self, quoteasset:str)->list:
        """ Returns a dict of the best X pairs per quoteasset based on their rolling-24H volume of quoteasset.
            Therefore, we can access the top trading pairs in terms of quoteasset volume and set our bots on them. """

        exchange = self.exchange

        pairs = exchange.GetNameOfPairs_WithQuoteasset(quoteAssets=[quoteasset])		# {'BTC':['ETHBTC', 'LTCBTC',...]}
        if pairs == {}:
            sys.exit("Could not get the pairs from Binance to sort them by day volume. Exiting the script.")

        stats = dict()
        self.skips = 0

        def GetVolume(pair:str):
            pair_info = exchange.GetMetadataOfPair(pair)

            if pair_info == {}:
                print("Could not get " + pair + " info from Binance to get its volume.")
                self.Skips()

            elif pair_info and pair_info['status'] == 'TRADING':
                stats[pair] = Decimal(self.exchange.Get24hrTicker(pair)['quoteVolume'])

        pool  = Pool()
        func1 = partial(GetVolume)
        pool.map(func1, pairs[quoteasset])
        pool.close()
        pool.join()

        sorted_volumes  = {k: v for k, v in sorted(stats.items(), key=lambda item: item[1], reverse=True)}
        sorted_pairs_list = list(sorted_volumes.keys())

        return sorted_pairs_list              # ['NPXSETH', 'DENTETH', 'HOTETH', 'KEYETH', 'NCASHETH', 'MFTETH', 'TRXETH', 'SCETH', 'ZILETH', 'STORMETH', ...]


    def SetTradingPairs(self)->bool:
        """ Gets the name of the pairs that we will be trading on
            and set the corresponding bots as active, if enough money in account. """

        sp       = yaspin()
        sp.color = 'green'
        exchange = self.exchange
        database = self.database
        BotsPerQuoteAsset = Settings.parameters['BotsPerQuoteAsset']

        trading_pairs = dict()
        sorted_pairs  = dict()
        # holding     = dict()

        open_positions = self.CheckOpenPositions()                                                              # Check for open positions (bots that are holding a coin)

        # List of all the quoteassets present in the database
        ExistingQuoteassets = list(set([dict(bot)['quoteasset'] for bot in database.GetAllBots()]))             # ['ETH', 'BTC']

        account_balance = dict()

        for quoteasset in ExistingQuoteassets:
            sp.text = "Processing " + quoteasset + " pairs..."
            sp.start()

            account_balance[quoteasset] = Decimal(database.GetAccountBalance(quoteasset=quoteasset, real_or_internal='internal'))

            # If we have room for more bots
            if len(open_positions[quoteasset]) < BotsPerQuoteAsset:
                # Find the best X pairs by volume
                sorted_pairs[quoteasset] = self.SortByDayVolume(quoteasset=quoteasset)                                                                                      # best_pairs = {'ETH' : ['HOTETH', 'DENTETH', 'NPXSETH', 'NCASHETH', 'KEYETH', 'ZILETH', 'TRXETH', 'SCETH', 'MFTETH', 'VETETH'],
                # Set the pairs to trade on for each quoteasset : the holding pairs, completed with the best pairs per volume if needed.                                    #               'BTC' : ['HOTBTC', 'VETBTC', 'ZILBTC', 'MBLBTC', 'MATICBTC', 'FTMBTC', 'TRXBTC', 'IOSTBTC', 'DOGEBTC', 'SCBTC']}
                # Do not trade on the quoteassets and BNB, to avoid balances problems.
                filtered = (k for k in sorted_pairs[quoteasset] if k not in open_positions[quoteasset] if not k.startswith(tuple([q for q in ExistingQuoteassets])) if not k.startswith('BNB'))
                trading_pairs[quoteasset] = open_positions[quoteasset] + list(islice(filtered, BotsPerQuoteAsset-len(open_positions[quoteasset])))

                min_amount = sum(Decimal(exchange.GetMinNotional(pair)) for pair in trading_pairs[quoteasset] if pair not in open_positions[quoteasset])
                # holding[quoteasset] = sum(Decimal(dict(bot)['quoteBalance']) for bot in database.GetAllBots() if dict(bot)['quoteasset']==quoteasset and dict(bot)['quoteBalance'] != '')           # How much bots are holding atm on this quoteasset.

                sp.stop()

                # Check if balance is enough to trade on the completing pairs
                if account_balance[quoteasset] > min_amount:
                    allocation = account_balance[quoteasset] / (len(trading_pairs[quoteasset])-len(open_positions[quoteasset]))
                    print("Paper trading on the top {nbpairs} pairs by volume on {quoteAsset} : {trading_pairs}".format(nbpairs=BotsPerQuoteAsset, quoteAsset=quoteasset, trading_pairs=trading_pairs[quoteasset]))
                    if open_positions[quoteasset]:
                        print("(using " + str(open_positions[quoteasset]) + " which are still trying to sell.)")
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
                print("No room left for new bots on " + quoteasset + ". Trading on " + str(open_positions[quoteasset]))

        if not [bot["status"] for bot in database.GetAllBots()]:
            print("\nYou can't trade on any quoteAsset. Please change the starting balance and try again ! \n_________________________________________________________________________________________________________________________")
            return False


    def WaitForCandle(self)->bool:
        """ Pauses the program until the next candle arrives.
            Just before the candle arrives, sets the pairs to trade on. """

        # self.SetTradingPairs()

        timeframe = Settings.parameters['timeframe']

        sp = yaspin()
        sp.start()

        parsed_timeframe = re.findall(r'[A-Za-z]+|\d+', timeframe)      # Separates '30m' in ['30', 'm']

        next_candle = None
        now = datetime.now()

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
        print(Fore.GREEN + "Candle : " + str(next_candle.strftime("%Y-%m-%d %H:%M")) + " (local time).")
        self.SetTradingPairs()          # Takes ~40secs to run

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
    def PrintTradeSequence(buys:dict, sells:dict):
        """ Prints the results of the last search for signals. """

        print("\n\t({nb_buys} Buys, {nb_sells} Sells.)".format(nb_buys=len(buys), nb_sells=len(sells)))
        print("")