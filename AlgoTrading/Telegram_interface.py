from Helpers import HelperMethods
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, ConversationHandler, CallbackQueryHandler
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
import configparser as cfg
from Exchange import Binance
from Database import BotDatabase
from decimal import Decimal
from datetime  import datetime


class TelegramInterface:

    def __init__(self, paper_trading:bool=True):
        self.paper_trading = paper_trading

        self.exchange = Binance(filename='assets/credentials.txt')
        self.database = BotDatabase(name="assets/database_paper.db") if self.paper_trading else BotDatabase(name="assets/database_live.db")
        self.helpers  = HelperMethods(database=self.database)

        # Get the quotes in the database
        unsorted_existing_quoteassets = [dict(bot)['quote'] for bot in self.database.get_all_bots()]
        self.existing_quoteassets = list(set(sorted(unsorted_existing_quoteassets)))                       # ['BTC', 'ETH']

        # Define a pattern to look for when clicking on a button
        quotes = '^('
        for quoteasset in self.existing_quoteassets:
            quotes = quotes + '|' + quoteasset
        self.quotes = quotes + ')$'

        # Get all the pairs we have traded on
        all_traded_pairs = sorted([dict(bot)['pair'] for bot in self. database.get_all_bots() if int(dict(bot)['number_of_orders']) >= 1])
        # Define a pattern to look for when clicking on a button
        pairs = '^('
        for pair_ in all_traded_pairs:
            pairs = pairs + '|' + pair_
        self.pairs = pairs + ')$'


    """ Menus _____________________________________________________________________________"""

    def Start(self, update, context):
        # Used only one time at the start of the bot, when using /start.
        update.message.reply_text(text         = 'Main menu:',
                                  reply_markup = self.main_menu_keyboard())

    def Main_Menu(self, update, context):
        query = update.callback_query
        query.answer()
        query.edit_message_text(text         = 'Main menu:',
                                reply_markup = self.main_menu_keyboard())

    def Select_Quote_Menu(self, update, context):
        query = update.callback_query
        query.answer()
        query.edit_message_text(text         = 'Select a quote:',
                                reply_markup = self.select_quote_menu_keyboard())

    def Select_Base_Menu(self, update, context):
        query = update.callback_query
        quote = query.data
        query.answer()
        query.edit_message_text(text         = f'Select a base to pair with {quote}:',
                                reply_markup = self.select_base_menu_keyboard(quote))



    """ Keyboards _________________________________________________________________________"""

    @staticmethod
    def main_menu_keyboard():

        keyboard = [[InlineKeyboardButton('GENERAL', callback_data='QuotesStats')], [InlineKeyboardButton('PAIRS', callback_data='Pair_stats')]]

        return InlineKeyboardMarkup(keyboard)

    def select_quote_menu_keyboard(self, ):
        # keyboard = [[InlineKeyboardButton('ETHBTC',         callback_data='ETHBTC')],
        #             [InlineKeyboardButton('Main menu',      callback_data='main')]]
        # Add a button for each quote in the database.
        quote_buttons = []
        for quote in self.existing_quoteassets:
            if list(self.database.get_quote_orders(quote=quote)):
                quote_buttons.append(InlineKeyboardButton(quote, callback_data=quote))

        keyboard = [quote_buttons, [InlineKeyboardButton('Main menu', callback_data='main')]]
        return InlineKeyboardMarkup(keyboard)

    def select_base_menu_keyboard(self, quote:str):

        pairs_ = sorted([dict(bot)['pair'] for bot in self.database.get_all_bots() if int(dict(bot)['number_of_orders']) >= 1 if dict(bot)['quote'] == quote])
        # bases_ = sorted([dict(bot)['pair'].replace(quote, '') for bot in self.database.get_all_bots() if int(dict(bot)['number_of_orders']) >= 1 if dict(bot)['quote'] == quote])
        bases_ = sorted([pair.replace(quote, '') for pair in pairs_])

        # Use bases_ for the display, pairs_ for the callback_data
        all_base_buttons = [InlineKeyboardButton(base, callback_data=pair) for base, pair in zip(bases_, pairs_)]

        # Group the buttons in pairs of 3, otherwise it doesn't fit on the screen.
        keyboard = [all_base_buttons[i:i+3] for i in range(0,len(all_base_buttons),3)]

        keyboard.append([InlineKeyboardButton('Main menu', callback_data='main')])
        return InlineKeyboardMarkup(keyboard)

    @staticmethod
    def return_menu_keyboard():
        keyboard = [[InlineKeyboardButton('Main menu',  callback_data='main')]]
        return InlineKeyboardMarkup(keyboard)



    """ Actions __________________________________________________________________________"""

    def QuotesStats(self, update, context):

        hold_duration = {}
        today = datetime.utcnow()
        text = """"""
        text = text + """```  """ + str(today.day) + """ """ + today.strftime('%h') + """ """ + today.strftime('%Y')+ """ """ + today.strftime("%H:%M:%S") + """   (UTC)```"""
        for quote in self.existing_quoteassets:
            if list(self.database.get_quote_orders(quote=quote)):
                hold_duration[quote] = self.helpers.quote_average_hold_duration(quote)

                text = text + """``` _______________________________ 
                 ``` *""" + quote + """ STATS :*"""
                text = text + """``` 
        All orders   = """ + "{total_orders} (+{recent_orders} in 24h)".format(total_orders=self.helpers.total_orders(quote), recent_orders=self.helpers.recent_orders(quote)) + """
        Open orders  = """ + str(self.helpers.open_orders(quote))  + """
        Binance bal. = """ + str(self.exchange.GetAccountBalance(quote).get('free'))  + """ """ + quote + """
        Intern bal.  = """ + str(self.database.get_db_account_balance(quote, internal_balance=True))  + """ """ + quote + """
        Profit       = """ + """{profit_in_quote} {quoteasset}
                       ({profit_in_percentage}%)""".format(quoteasset           = quote,
                                                           profit_in_quote      = self.database.get_db_account_balance(quote, internal_profit=True),
                                                           profit_in_percentage = format(round(Decimal(self.database.get_db_account_balance(quote, internal_profit=True))/Decimal(self.database.get_db_account_balance(quote, started_with=True))*100, 2), 'f')) + """
        Fees         = """ + """{fees_in_quote} {quoteasset}
                       ({fees_in_BNB} BNB)""".format(quoteasset    = quote,
                                                     fees_in_quote = self.database.get_db_account_balance(quote, internal_quote_fees=True),
                                                     fees_in_BNB   = format(round(Decimal(self.database.get_db_account_balance(quote, internal_BNB_fees=True)), 3), 'f')) + """ 
        Profit-Fees  = """ + """{profit_minus_fees_in_quote} {quoteasset}
                       ({profit_minus_fees_in_quote_in_percentage}%)""".format(quoteasset                 = quote,
                                                                               profit_minus_fees_in_quote = self.database.get_db_account_balance(quote, internal_profit_minus_fees=True),
                                                                               profit_minus_fees_in_quote_in_percentage = format(round(Decimal(self.database.get_db_account_balance(quote, internal_profit_minus_fees=True))/Decimal(self.database.get_db_account_balance(quote=quote, started_with=True))*100, 2), 'f')) + """
        Hold dur.    = """ + """{days}d, {hours}h, {minutes}m, {seconds}s""".format(days       = hold_duration[quote].days,
                                                                                    hours      = hold_duration[quote].days * 24 + hold_duration[quote].seconds // 3600,
                                                                                    minutes    = (hold_duration[quote].seconds % 3600) // 60,
                                                                                    seconds    = hold_duration[quote].seconds % 60) + """
        
        ```"""

        query = update.callback_query
        query.answer()
        query.edit_message_text(text         = text,
                                parse_mode   = "Markdown",
                                reply_markup = self.return_menu_keyboard())

    def PairStats(self, update, context):

        query = update.callback_query
        pair = query.data

        quote = ''
        for quote_ in self.existing_quoteassets:
            if quote_ in pair:
                quote = quote_

        # Check if the bot sold at least one time and adapt what's displayed
        if len(list(self.database.get_orders_of_bot(pair))) > 1:
            profit = f"{dict(self.database.get_bot(pair=pair))['bot_profit']} {quote}"

            profit_minus_fees = "{profit_minus_fees} {quoteasset}".format(quoteasset        = quote,
                                                                          profit_minus_fees = dict(self.database.get_bot(pair=pair))['bot_profit_minus_fees'])

            average_hold_duration_string = "{days}d, {hours}h, {minutes}m, {seconds}s.".format(days       = self.helpers.pair_average_hold_duration(pair).days,
                                                                                               hours      = self.helpers.pair_average_hold_duration(pair).days * 24 + self.helpers.pair_average_hold_duration(pair).seconds // 3600,
                                                                                               minutes    = (self.helpers.pair_average_hold_duration(pair).seconds % 3600) // 60,
                                                                                               seconds    = self.helpers.pair_average_hold_duration(pair).seconds % 60)
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
    All orders  = """ + "{pair_total_orders} (+{recent_orders} in 24h)".format(pair_total_orders=len(list(self.database.get_orders_of_bot(pair))), recent_orders=self.helpers.pair_recent_orders(pair)) + """
    Profit      = """ + profit + """
    Fees        = """ + """{fees_in_quote} {quoteasset}
                  ({fees_in_BNB} BNB)""".format(quoteasset    = quote,
                                                fees_in_quote = dict(self.database.get_bot(pair=pair))['bot_quote_fees'],
                                                fees_in_BNB   = format(round(Decimal(dict(self.database.get_bot(pair=pair))['bot_BNB_fees']), 3), 'f')) + """
    Profit-Fees = """ + profit_minus_fees + """
    Hold dur.   = """ + average_hold_duration_string + """
    ```"""

        query.answer()
        query.edit_message_text(text         = text,
                                parse_mode   = "Markdown",
                                reply_markup = self.return_menu_keyboard())



    """ Main ______________________________________________________________________________"""
    @staticmethod
    def read_token_from_config_file(config):
        parser = cfg.ConfigParser()
        parser.read(config)
        return parser.get('creds', 'token')

    def start_telegram_bot(self):

        updater = Updater(self.read_token_from_config_file("assets/telegram_config.cfg"), use_context=True)

        updater.dispatcher.add_handler(CommandHandler('start', self.Start))
        updater.dispatcher.add_handler(CallbackQueryHandler(self.Main_Menu,              pattern='main'))
        updater.dispatcher.add_handler(CallbackQueryHandler(self.QuotesStats,            pattern='QuotesStats'))
        updater.dispatcher.add_handler(CallbackQueryHandler(self.Select_Quote_Menu,      pattern='Pair_stats'))
        updater.dispatcher.add_handler(CallbackQueryHandler(self.Select_Base_Menu,       pattern=self.quotes))
        updater.dispatcher.add_handler(CallbackQueryHandler(self.PairStats,              pattern=self.pairs))

        updater.start_polling()


if __name__ == '__main__':
    TelegramInterface(paper_trading=True).start_telegram_bot()