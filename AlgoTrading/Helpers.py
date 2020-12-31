from   Exchange import Binance
from   Database import BotDatabase
import datetime
from   datetime import datetime, timedelta


exchange = Binance(filename='assets/credentials.txt')
database = BotDatabase(name="assets/database.db")


class HelperMethods:
    def __init__(self):
        pass

    # ____________________________ QUOTEASSETS ____________________________
    @staticmethod
    def total_orders(quote:str)->int:
        """ Returns the number of all the orders ever made on a quote. """
        return len(list(database.GetQuoteassetOrders(quoteasset=quote)))

    @staticmethod
    def recent_orders(quote:str)->int:
        """ Returns the number of orders we did in the last 24 hours on a quote. """
        return len([dict(order)['pair'] for order in list(database.GetQuoteassetOrders(quote))
                    if datetime.utcnow()-timedelta(hours=24) < datetime.strptime(dict(order)['transactTime'], "%Y-%m-%d %H:%M:%S")])

    @staticmethod
    def open_orders(quote:str)->int:
        """ Returns the number of currently open buy positions. """
        return len([dict(bot)['pair'] for bot in database.GetAllBots() if dict(bot)['status']=='Looking to exit' if dict(bot)['quoteasset']==quote])

    @staticmethod
    def quote_average_hold_duration(quote:str):
        """Computes the average holding duration on a quoteasset."""

        hold_durations = []
        for order in list(database.GetQuoteassetOrders(quoteasset=quote)):
            if dict(order)['side'] == 'SELL':
                lst = dict(order)['hold_duration'].split(":")
                if len(lst[0])<=2:
                    hold_durations.append(timedelta(hours=int(lst[0]), minutes=int(lst[1]), seconds=int(lst[2])))
                else:
                    temp  = lst[0].split(" ")
                    days  = temp[0]
                    hours = temp[-1]
                    hold_durations.append(timedelta(days=int(days), hours=int(hours), minutes=int(lst[1]), seconds=int(lst[2])))

        # for item in hold_durations:
        #     print(item)

        td = sum(hold_durations, timedelta()) / len(hold_durations)		# Average of the timedeltas
        # print("Average holding duration on {quote} : {days}d, {hours}h, {minutes}m, {seconds}s.".format(quote = quote,
        #                                                                                                      days       = td.days,
        #                                                                                                      hours      = td.days * 24 + td.seconds // 3600,
        #                                                                                                      minutes    = (td.seconds % 3600) // 60,
        #                                                                                                      seconds    = td.seconds % 60))
        return td

    @staticmethod
    def pair_recent_orders(pair:str)->int:
        """ Returns the number of orders we did in the last 24 hours on a pair. """
        return len([dict(order)['pair'] for order in list(database.GetOrdersOfBot(pair))
                    if datetime.utcnow()-timedelta(hours=24) < datetime.strptime(dict(order)['transactTime'], "%Y-%m-%d %H:%M:%S")])

    @staticmethod
    def pair_average_hold_duration(pair:str):
        """ Computes the average hold duration on a quoteasset."""

        hold_durations = []
        ordersOfBot = list(database.GetOrdersOfBot(pair=pair))

        if len(ordersOfBot) > 1:
            for order in ordersOfBot:
                if dict(order)['side'] == 'SELL':
                    lst = dict(order)['hold_duration'].split(":")
                    if len(lst[0])<=2:
                        # If we holded less than a day
                        hold_durations.append(timedelta(hours=int(lst[0]), minutes=int(lst[1]), seconds=int(lst[2])))
                    else:
                        # If we holded for more than a day
                        temp  = lst[0].split(" ")
                        days  = temp[0]
                        hours = temp[-1]
                        hold_durations.append(timedelta(days=int(days), hours=int(hours), minutes=int(lst[1]), seconds=int(lst[2])))

            td = sum(hold_durations, timedelta()) / len(hold_durations)		# Average of the timedeltas

            return td
        else:
            # If the bot did only 1 buy order and never sold, we can't compute the hold_duration!
            return None
