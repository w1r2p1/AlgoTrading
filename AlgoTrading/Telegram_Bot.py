from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, ConversationHandler, CallbackQueryHandler
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
import telegram
import configparser as cfg
from Exchange import Binance
from Database import BotDatabase
import requests
import json
from decimal import Decimal
import numpy as np
from functools import wraps
from datetime  import datetime
import calendar

from Dashboard_live import TotalOrders, OpenOrders, Quoteasset_AverageHoldDuration   # , Profit, QuoteFees, BNBFees, ProfitMinusFees
from Dashboard_live import Pair_AverageHoldDuration, RecentOrders, Pair_RecentOrders


def read_token_from_config_file(config):
    parser = cfg.ConfigParser()
    parser.read(config)
    return parser.get('creds', 'token')


exchange = Binance(filename='credentials.txt')
database = BotDatabase("database.db")

# Get the quotes in the database
Unsorted_ExistingQuoteassets = [dict(bot)['quoteasset'] for bot in database.GetAllBots()]
ExistingQuoteassets = list(set(sorted(Unsorted_ExistingQuoteassets)))                       # ['BTC', 'ETH']

# Define a pattern to look for when clicking on a button
quotes = '^('
for quoteasset in ExistingQuoteassets:
    quotes = quotes + '|' + quoteasset
quotes = quotes + ')$'

# Get all the pairs we have traded on
all_traded_pairs = [dict(bot)['pair'] for bot in database.GetAllBots() if int(dict(bot)['number_of_orders'])>=1]
# Define a pattern to look for when clicking on a button
pairs = '^('
for pair in all_traded_pairs:
    pairs = pairs + '|' + pair
pairs = pairs + ')$'




""" Menus _____________________________________________________________________________"""

def Start(update, context):
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

def Select_Pair_In_Quote_Menu(update, context):
    query = update.callback_query
    quote = query.data
    query.answer()
    query.edit_message_text(text         = 'Select a pair in ' + quote + ' :',
                            reply_markup = select_pair_in_quote_menu_keyboard(quote))



""" Keyboards _________________________________________________________________________"""

def main_menu_keyboard():

    keyboard = [[InlineKeyboardButton('QUOTES', callback_data='QuotesStats')], [InlineKeyboardButton('PAIRS', callback_data='Pair_stats')]]

    return InlineKeyboardMarkup(keyboard)

def select_quote_menu_keyboard():
    # keyboard = [[InlineKeyboardButton('ETHBTC',         callback_data='ETHBTC')],
    #             [InlineKeyboardButton('Main menu',      callback_data='main')]]
    # Add a button for each quoteasset in the database.
    quote_buttons = []
    for quote in ExistingQuoteassets:
        quote_buttons.append(InlineKeyboardButton(quote, callback_data=quote))

    keyboard = [quote_buttons, [InlineKeyboardButton('Main menu', callback_data='main')]]
    return InlineKeyboardMarkup(keyboard)

def select_pair_in_quote_menu_keyboard(quote:str):

    unsorted_pairs    = [dict(bot)['pair'].replace(quote, '') for bot in database.GetAllBots() if int(dict(bot)['number_of_orders'])>=1 if dict(bot)['quoteasset']==quote]
    sorted_pairs      = sorted(unsorted_pairs)
    all_pairs_buttons = [InlineKeyboardButton(pair, callback_data=pair) for pair in sorted_pairs]

    # Group the buttons in pairs of 3, otherwise it doesn't fit on the screen.
    keyboard = [all_pairs_buttons[i:i+3] for i in range(0,len(all_pairs_buttons),3)]

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
    for quote in ExistingQuoteassets:
        text = text + """``` _______________________________ 
         ``` *""" + quote + """ STATS :*"""
        text = text + """``` 

All orders   = """ + "{total_orders} (+{recent_orders} in 24h)".format(total_orders=TotalOrders(quote), recent_orders=RecentOrders(quote)) + """
Open orders  = """ + str(OpenOrders(quote))  + """
Binance bal. = """ + str(exchange.GetAccountBalance(quote))  + """ """ + quote + """
Bot bal.     = """ + str(database.GetAccountBalance(quote, real_or_internal='internal'))  + """ """ + quote + """
Profit       = """ + """{profit_in_quote} {quoteasset}
               ({profit_in_percentage}%)""".format(quoteasset           = quote,
                                                   profit_in_quote      = database.GetProfit(quote),
                                                   profit_in_percentage = format(round(Decimal(database.GetProfit(quote))/Decimal(database.GetStartBalance(quote))*100, 2), 'f')) + """
Fees         = """ + """{fees_in_quote} {quoteasset}
               ({fees_in_BNB} BNB)""".format(quoteasset    = quoteasset,
                                             fees_in_quote = database.GetQuoteFees(quote),
                                             fees_in_BNB   = format(round(Decimal(database.GetBNBFees(quote)), 3), 'f')) + """ 
Profit-Fees  = """ + """{profit_minus_fees_in_quote} {quoteasset}
               ({profit_minus_fees_in_quote_in_percentage}%)""".format(quoteasset                 = quote,
                                                                       profit_minus_fees_in_quote = database.GetProfit_minus_fees(quote),
                                                                       profit_minus_fees_in_quote_in_percentage = format(round(Decimal(database.GetProfit_minus_fees(quote))/Decimal(database.GetStartBalance(quoteasset=quote))*100, 2), 'f')) + """
Hold dur.    = """ + """{days}d, {hours}h, {minutes}m, {seconds}s""".format(days       = Quoteasset_AverageHoldDuration(quote).days,
                                                                            hours      = Quoteasset_AverageHoldDuration(quote).days * 24 + Quoteasset_AverageHoldDuration(quote).seconds // 3600,
                                                                            minutes    = (Quoteasset_AverageHoldDuration(quote).seconds % 3600) // 60,
                                                                            seconds    = Quoteasset_AverageHoldDuration(quote).seconds % 60) + """

```"""

    query = update.callback_query
    query.answer()
    query.edit_message_text(text         = text,
                            parse_mode   = "Markdown",
                            reply_markup = return_menu_keyboard())


def PairStats(update, context):

    query = update.callback_query
    pair = query.data

    pair_quote = ''
    for quote in ExistingQuoteassets:
        if quote in pair:
            pair_quote = quote

    # Check if the bot sold at least one time and adapt what's displayed
    if len(list(database.GetOrdersOfBot(pair))) > 1:
        profit = dict(database.GetBot(pair=pair))['bot_profit'] + " " + pair_quote

        profit_minus_fees = "{profit_minus_fees} {quoteasset}".format(quoteasset        = pair_quote,
                                                                      profit_minus_fees = dict(database.GetBot(pair=pair))['bot_profit_minus_fees'])

        average_hold_duration_string = "{days}d, {hours}h, {minutes}m, {seconds}s.".format(days       = Pair_AverageHoldDuration(pair).days,
                                                                                           hours      = Pair_AverageHoldDuration(pair).days * 24 + Pair_AverageHoldDuration(pair).seconds // 3600,
                                                                                           minutes    = (Pair_AverageHoldDuration(pair).seconds % 3600) // 60,
                                                                                           seconds    = Pair_AverageHoldDuration(pair).seconds % 60)
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

All orders  = """ + "{pair_total_orders} (+{recent_orders} in 24h)".format(pair_total_orders=len(list(database.GetOrdersOfBot(pair))), recent_orders=Pair_RecentOrders(pair)) + """
Profit      = """ + profit + """
Fees        = """ + """{fees_in_quote} {quoteasset}
              ({fees_in_BNB} BNB)""".format(quoteasset    = pair_quote,
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

def main():

    updater = Updater(read_token_from_config_file("config.cfg"), use_context=True)

    updater.dispatcher.add_handler(CommandHandler('start', Start))
    updater.dispatcher.add_handler(CallbackQueryHandler(Main_Menu,                  pattern='main'))
    updater.dispatcher.add_handler(CallbackQueryHandler(QuotesStats,                pattern='QuotesStats'))
    updater.dispatcher.add_handler(CallbackQueryHandler(Select_Quote_Menu,          pattern='Pair_stats'))
    updater.dispatcher.add_handler(CallbackQueryHandler(Select_Pair_In_Quote_Menu,  pattern=quotes))
    updater.dispatcher.add_handler(CallbackQueryHandler(PairStats,                  pattern=pairs))


    updater.start_polling()


if __name__ == '__main__':
    main()