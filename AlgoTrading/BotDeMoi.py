from RealTrading  import RealTrading
from PaperTrading import PaperTrading
from Exchange     import Binance
from Database     import BotDatabase
from BackTesting  import BackTesting
from BotsCreation import BotsCreation
import Settings

answer = 'e'

# Initialization    : Create a bot on all the pairs trading on a quoteAsset.
# Trading           : Allow trading only on the best 10 pairs by volume (which are currently tradable), mark the other bots as inactive.
#                     Each time a bot sells, check if the pair is still in the best pairs list. If yes, continue to trade, if not, mark it as inactive and activate another bot.

def Main():

    # Clear file database.db
    # with open('database.db', "w"):
    #     pass

    exchange        = Binance(filename = 'credentials.txt')
    database        = BotDatabase("database.db")
    botsCreator     = BotsCreation(exchange, database)
    realtrader      = RealTrading(exchange,  database)
    papertrader     = PaperTrading(exchange, database)
    backtester      = BackTesting(exchange, Settings.parameters['strategy'])


    assert Settings.parameters['timeframe'] in exchange.KLINE_INTERVALS, Settings.parameters['timeframe'] + " is not a valid interval."

    # Welcome message
    # answer = None
    # while answer not in ['b', 'e', 'q']:
    #     answer = input('Welcome to BotDeMoi ! Execute (e) or Backtest (b) the strategy on all the pairs ?\n')


    if   answer == 'b':

        print("_______________________________________________________________________________________________")
        print("Entering Backtesting mode.\n")

        backtester.BacktestAllPairs(
                                    # pairs	    = exchange.GetNameOfPairs_WithQuoteasset(quoteAssets=Settings.parameters['WantedQuoteAssets']),     # {'BTC':['ETHBTC']}
                                    pairs	    = {'BTC':['ADABTC']},
                                    # timeframe	= Settings.parameters['timeframe'],
                                    plot_data   = True)


    elif answer == 'e':

        print("_______________________________________________________________________________________________")
        print("Entering Trading mode.\n")

        # Delete the bots on a specific quoteAsset.
        database.DeleteBots(quoteasset='')

        # Create and save the bots in the db that don't already exist.
        botsCreator.CreateAllBots(parameters=Settings.parameters)

        # Start (paper)trading.
        if Settings.parameters['test_orders']:
            papertrader.StartExecution()
        else:
            realtrader.StartExecution()



if __name__ == "__main__":
    Main()