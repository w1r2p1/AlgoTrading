# This files stores the settings that will be used across the project.
# You need to have BNB in your Binance account : we will use that to pay the fees and save 25% on them (0.1% -> 0.075%).

# Necessary modules :
# pandas
# keyboard
# yaspin


parameters = dict(
                WantedQuoteAssets            = ['BTC', 'ETH'],
                BotsPerQuoteAsset            = 10,
                strategy                     = 'maCrossoverStrategy',
                timeframe                    = '5m',
                test_orders                  = True,                            # Note that Test orders always return blank responses.
                start_balance_if_test_orders = 3,                               # All the quoteassets will start at the same balance
                )