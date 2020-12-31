from Helpers import HelperMethods
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, ConversationHandler, CallbackQueryHandler
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
import configparser as cfg
from Exchange import Binance
from Database import BotDatabase
from decimal import Decimal
from datetime  import datetime

def read_token_from_config_file(config):
    parser = cfg.ConfigParser()
    parser.read(config)
    return parser.get('creds', 'token')


exchange = Binance(filename='assets/credentials.txt')
database = BotDatabase(name="assets/database.db")
helpers  = HelperMethods()

# Get the quotes in the database
unsorted_existing_quoteassets = [dict(bot)['quoteasset'] for bot in database.GetAllBots()]
existing_quoteassets = list(set(sorted(unsorted_existing_quoteassets)))                       # ['BTC', 'ETH']

# Define a pattern to look for when clicking on a button
quotes = '^('
for quoteasset in existing_quoteassets:
    quotes = quotes + '|' + quoteasset
quotes = quotes + ')$'

# Get all the pairs we have traded on
all_traded_pairs = sorted([dict(bot)['pair'] for bot in database.GetAllBots() if int(dict(bot)['number_of_orders'])>=1])
# Define a pattern to look for when clicking on a button
pairs = '^('
for pair_ in all_traded_pairs:
    pairs = pairs + '|' + pair_
pairs = pairs + ')$'


""" Menus _____________________________________________________________________________"""

def Start(update, context):
    # Used only one time at the start of the bot, when using /start.
    update.message.reply_text(text         = 'Main menu:',
                              reply_markup = main_menu_keyboard())

def Main_Menu(update, context):
    query = update.callback_query
    query.answer()
    query.edit_message_text(text         = 'Main menu:',
                            reply_markup = main_menu_keyboard())

def Select_Quote_Menu(update, context):
    query = update.callback_query
    query.answer()
    query.edit_message_text(text         = 'Select a quote:',
                            reply_markup = select_quote_menu_keyboard())

def Select_Base_Menu(update, context):
    query = update.callback_query
    quote = query.data
    query.answer()
    query.edit_message_text(text         = f'Select a base to pair with {quote}:',
                            reply_markup = select_base_menu_keyboard(quote))



""" Keyboards _________________________________________________________________________"""

def main_menu_keyboard():

    keyboard = [[InlineKeyboardButton('GENERAL', callback_data='QuotesStats')], [InlineKeyboardButton('PAIRS', callback_data='Pair_stats')]]

    return InlineKeyboardMarkup(keyboard)

def select_quote_menu_keyboard():
    # keyboard = [[InlineKeyboardButton('ETHBTC',         callback_data='ETHBTC')],
    #             [InlineKeyboardButton('Main menu',      callback_data='main')]]
    # Add a button for each quote in the database.
    quote_buttons = []
    for quote in existing_quoteassets:
        quote_buttons.append(InlineKeyboardButton(quote, callback_data=quote))

    keyboard = [quote_buttons, [InlineKeyboardButton('Main menu', callback_data='main')]]
    return InlineKeyboardMarkup(keyboard)

def select_base_menu_keyboard(quote:str):

    pairs_ = sorted([dict(bot)['pair'] for bot in database.GetAllBots() if int(dict(bot)['number_of_orders'])>=1 if dict(bot)['quoteasset']==quote])
    bases_ = sorted([dict(bot)['pair'].replace(quote, '') for bot in database.GetAllBots() if int(dict(bot)['number_of_orders'])>=1 if dict(bot)['quoteasset']==quote])

    # Use bases_ for the display, pairs_ for the callback_data
    all_base_buttons = [InlineKeyboardButton(base, callback_data=pair) for base, pair in zip(bases_, pairs_)]

    # Group the buttons in pairs of 3, otherwise it doesn't fit on the screen.
    keyboard = [all_base_buttons[i:i+3] for i in range(0,len(all_base_buttons),3)]

    keyboard.append([InlineKeyboardButton('Main menu', callback_data='main')])
    return InlineKeyboardMarkup(keyboard)

def return_menu_keyboard():
    keyboard = [[InlineKeyboardButton('Main menu',  callback_data='main')]]
    return InlineKeyboardMarkup(keyboard)



""" Actions __________________________________________________________________________"""

def QuotesStats(update, context):

    today = datetime.utcnow()
    text = """"""
    text = text + """```  """ + str(today.day) + """ """ + today.strftime('%h') + """ """ + today.strftime('%Y')+ """ """ + today.strftime("%H:%M:%S") + """   (UTC)```"""
    for quote in existing_quoteassets:
        text = text + """``` _______________________________ 
         ``` *""" + quote + """ STATS :*"""
        text = text + """``` 

All orders   = """ + "{total_orders} (+{recent_orders} in 24h)".format(total_orders=helpers.total_orders(quote), recent_orders=helpers.recent_orders(quote)) + """
Open orders  = """ + str(helpers.open_orders(quote))  + """
Binance bal. = """ + str(exchange.GetAccountBalance(quote))  + """ """ + quote + """
Intern bal.  = """ + str(database.GetAccountBalance(quote, real_or_internal='internal'))  + """ """ + quote + """
Profit       = """ + """{profit_in_quote} {quoteasset}
               ({profit_in_percentage}%)""".format(quoteasset           = quote,
                                                   profit_in_quote      = database.get_profit(quote, 'internal'),
                                                   profit_in_percentage = format(round(Decimal(database.get_profit(quote, 'internal'))/Decimal(database.GetStartBalance(quote))*100, 2), 'f')) + """
Fees         = """ + """{fees_in_quote} {quoteasset}
               ({fees_in_BNB} BNB)""".format(quoteasset    = quoteasset,
                                             fees_in_quote = database.GetQuoteFees(quote),
                                             fees_in_BNB   = format(round(Decimal(database.GetBNBFees(quote)), 3), 'f')) + """ 
Profit-Fees  = """ + """{profit_minus_fees_in_quote} {quoteasset}
               ({profit_minus_fees_in_quote_in_percentage}%)""".format(quoteasset                 = quote,
                                                                       profit_minus_fees_in_quote = database.GetProfit_minus_fees(quote),
                                                                       profit_minus_fees_in_quote_in_percentage = format(round(Decimal(database.GetProfit_minus_fees(quote))/Decimal(database.GetStartBalance(quoteasset=quote))*100, 2), 'f')) + """
Hold dur.    = """ + """{days}d, {hours}h, {minutes}m, {seconds}s""".format(days       = helpers.quote_average_hold_duration(quote).days,
                                                                            hours      = helpers.quote_average_hold_duration(quote).days * 24 + helpers.quote_average_hold_duration(quote).seconds // 3600,
                                                                            minutes    = (helpers.quote_average_hold_duration(quote).seconds % 3600) // 60,
                                                                            seconds    = helpers.quote_average_hold_duration(quote).seconds % 60) + """

```"""

    query = update.callback_query
    query.answer()
    query.edit_message_text(text         = text,
                            parse_mode   = "Markdown",
                            reply_markup = return_menu_keyboard())

def PairStats(update, context):

    query = update.callback_query
    pair = query.data

    quote = ''
    for quote_ in existing_quoteassets:
        if quote_ in pair:
            quote = quote_

    # Check if the bot sold at least one time and adapt what's displayed
    if len(list(database.GetOrdersOfBot(pair))) > 1:
        profit = f"{dict(database.GetBot(pair=pair))['bot_profit']} {quote}"

        profit_minus_fees = "{profit_minus_fees} {quoteasset}".format(quoteasset        = quote,
                                                                      profit_minus_fees = dict(database.GetBot(pair=pair))['bot_profit_minus_fees'])

        average_hold_duration_string = "{days}d, {hours}h, {minutes}m, {seconds}s.".format(days       = helpers.pair_average_hold_duration(pair).days,
                                                                                           hours      = helpers.pair_average_hold_duration(pair).days * 24 + helpers.pair_average_hold_duration(pair).seconds // 3600,
                                                                                           minutes    = (helpers.pair_average_hold_duration(pair).seconds % 3600) // 60,
                                                                                           seconds    = helpers.pair_average_hold_duration(pair).seconds % 60)
    else:
        profit                       = "Didn't sell yet"
        profit_minus_fees            = "Didn't sell yet"
        average_hold_duration_string = "Didn't sell yet"


    today = datetime.utcnow()
    text = """"""
    text = text + """```  """ + str(today.day) + """ """ + today.strftime('%h') + """ """ + today.strftime('%Y')+ """ """ + today.strftime("%H:%M:%S") + """   (UTC)```"""
    text = text + """``` _______________________________ 
         ``` *""" + pair + """ STATS :*"""
    text = text + """``` 

All orders  = """ + "{pair_total_orders} (+{recent_orders} in 24h)".format(pair_total_orders=len(list(database.GetOrdersOfBot(pair))), recent_orders=helpers.pair_recent_orders(pair)) + """
Profit      = """ + profit + """
Fees        = """ + """{fees_in_quote} {quoteasset}
              ({fees_in_BNB} BNB)""".format(quoteasset    = quote,
                                            fees_in_quote = dict(database.GetBot(pair=pair))['bot_quote_fees'],
                                            fees_in_BNB   = format(round(Decimal(dict(database.GetBot(pair=pair))['bot_BNB_fees']), 3), 'f')) + """
Profit-Fees = """ + profit_minus_fees + """
Hold dur.   = """ + average_hold_duration_string + """
```"""

    query.answer()
    query.edit_message_text(text         = text,
                            parse_mode   = "Markdown",
                            reply_markup = return_menu_keyboard())



""" Main ______________________________________________________________________________"""

def start_telegram_bot():

    updater = Updater(read_token_from_config_file("assets/telegram_config.cfg"), use_context=True)

    updater.dispatcher.add_handler(CommandHandler('start', Start))
    updater.dispatcher.add_handler(CallbackQueryHandler(Main_Menu,              pattern='main'))
    updater.dispatcher.add_handler(CallbackQueryHandler(QuotesStats,            pattern='QuotesStats'))
    updater.dispatcher.add_handler(CallbackQueryHandler(Select_Quote_Menu,      pattern='Pair_stats'))
    updater.dispatcher.add_handler(CallbackQueryHandler(Select_Base_Menu,       pattern=quotes))
    updater.dispatcher.add_handler(CallbackQueryHandler(PairStats,              pattern=pairs))

    updater.start_polling()


if __name__ == '__main__':
    start_telegram_bot()