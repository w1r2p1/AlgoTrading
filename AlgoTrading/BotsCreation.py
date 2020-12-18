from yaspin  import yaspin
from multiprocessing.pool import ThreadPool as Pool
from functools import partial
from datetime import datetime
import sys

class BotsCreation:

    def __init__(self, exchange, database):
        self.exchange = exchange        # GetNameOfPairs_WithQuoteasset
        self.database = database        # Savee the bots to the db


    def CreateBot(self, pair:str, quoteasset:str, strategy:str, timeframe:str, test_orders:bool):
        """ Creates and saves a bot with a set of parameters. """

        exchange = self.exchange

        # Get the quotePrecision for this pair
        pair_info = exchange.GetMetadataOfPair(pair)
        BNB_info  = exchange.GetMetadataOfPair(pair='BNB'+quoteasset)

        if pair_info:

            # Create the bot with a set of parameters
            bot = dict(pair				     = pair,
                       quoteasset		     = quoteasset,
                       strategy			     = strategy,
                       timeframe 		     = timeframe,
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
                       BNB_precision         = int(BNB_info['baseAssetPrecision']),
                       test_orders 		     = test_orders)

            self.database.SaveBot(bot)											# Each row of the table 'bots' is made of the values of 'bot'.


    def CreateAllBots(self, parameters:dict):
        """ Creates all the possible bots on each quoteAsset if they don't exist. """

        database = self.database
        exchange = self.exchange
        sp       = yaspin()
        sp.color = 'green'

        # List of all the quoteassets present in the database
        ExistingQuoteassets = set([dict(bot)['quoteasset'] for bot in database.GetAllBots()])                           # ['ETH', 'BTC']
        # Unique values of the two. | = union of 2 sets.
        AllQuoteassets = list(ExistingQuoteassets | set(parameters['WantedQuoteAssets']))                               # ['ETH', 'BTC']

        pairs  = exchange.GetNameOfPairs_WithQuoteasset(AllQuoteassets)                                                 # {'ETH': ['QTUMETH', 'EOSETH',..], 'BTC':[]}

        if pairs == {}:
            sys.exit("Could not get the pairs from Binance to create the bots. Exiting the script.")

        for quoteasset in AllQuoteassets:
            sp.text = "Creating the bots on " + quoteasset
            sp.start()
            # Create the bots that don't already exist in parallel. Use the default number of workers in Pool(), which given by os.cpu_count(). Here, 8 are used.
            pool  = Pool()
            func1 = partial(self.CreateBot, quoteasset=quoteasset, strategy=parameters['strategy'], timeframe=parameters['timeframe'], test_orders=parameters['test_orders'])
            pairs_without_bot = [pair for pair in pairs[quoteasset] if not self.database.GetBot(pair)]
            pool.map(func1, pairs_without_bot)
            pool.close()
            pool.join()
            sp.stop()
            if quoteasset in parameters['WantedQuoteAssets'] and quoteasset in ExistingQuoteassets:
                print("You want to trade on {quoteasset}, which is already in the database. {created_bots} new pairs available. Total = {total}.".format(quoteasset=quoteasset, created_bots=len(pairs_without_bot), total=len(pairs[quoteasset])))
            elif quoteasset in parameters['WantedQuoteAssets']:
                print("You want to trade on {quoteasset}, which is not in the database. Created {created_bots} bots.".format(quoteasset=quoteasset, created_bots=len(pairs_without_bot)))
            else:
                print("Note that {quoteasset} is already existing in the database. {created_bots} new pairs available. Total = {total}.".format(quoteasset=quoteasset, created_bots=len(pairs_without_bot), total=len(pairs[quoteasset])))