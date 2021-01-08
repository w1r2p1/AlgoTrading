import datetime
from   datetime import datetime, timedelta


class HelperMethods:
    def __init__(self, database):
        self.database = database

    def total_orders(self, quote:str)->int:
        """ Returns the number of all the orders ever made on a quote. """
        return len(list(self.database.get_quote_orders(quote=quote)))

    def recent_orders(self, quote:str)->int:
        """ Returns the number of orders we did in the last 24 hours on a quote. """
        return len([dict(order)['pair'] for order in list(self.database.get_quote_orders(quote))
                    if datetime.utcnow()-timedelta(hours=24) < datetime.strptime(dict(order)['transactTime'], "%Y-%m-%d %H:%M:%S")])

    def open_orders(self, quote:str)->int:
        """ Returns the number of currently open buy positions. """
        return len([dict(bot)['pair'] for bot in self.database.GetAllBots() if dict(bot)['status']=='Looking to exit' if dict(bot)['quote']==quote])

    def quote_average_hold_duration(self, quote:str):
        """Computes the average holding duration on a quote."""

        hold_durations = []
        for order in list(self.database.get_quote_orders(quote=quote)):
            if dict(order)['side'] == 'SELL':
                lst = dict(order)['hold_duration']
                if '-' not in lst:
                    lst = lst.split(":")

                    # Look for milliseconds
                    milliseconds = 0
                    split_milli = lst[2].split('.')
                    if len(split_milli)==2:
                        milliseconds = split_milli[1]

                    # Create a list of timedeltas
                    if len(lst[0])<=2:
                        hold_durations.append(timedelta(hours=int(lst[0]), minutes=int(lst[1]), seconds=int(lst[2]), milliseconds=milliseconds))
                    else:
                        temp  = lst[0].split(" ")
                        days  = temp[0]
                        hours = temp[-1]
                        hold_durations.append(timedelta(days=int(days), hours=int(hours), minutes=int(lst[1]), seconds=int(lst[2]), milliseconds=milliseconds))

        # for item in hold_durations:
        #     print(item)

        td = sum(hold_durations, timedelta()) / len(hold_durations)		# Average of the timedeltas
        # print("Average holding duration on {quote} : {days}d, {hours}h, {minutes}m, {seconds}s.".format(quote = quote,
        #                                                                                                      days       = td.days,
        #                                                                                                      hours      = td.days * 24 + td.seconds // 3600,
        #                                                                                                      minutes    = (td.seconds % 3600) // 60,
        #                                                                                                      seconds    = td.seconds % 60))
        return td

    def pair_recent_orders(self, pair:str)->int:
        """ Returns the number of orders we did in the last 24 hours on a pair. """
        return len([dict(order)['pair'] for order in list(self.database.GetOrdersOfBot(pair))
                    if datetime.utcnow()-timedelta(hours=24) < datetime.strptime(dict(order)['transactTime'], "%Y-%m-%d %H:%M:%S")])

    def pair_average_hold_duration(self, pair:str):
        """ Computes the average hold duration on a quote."""

        hold_durations = []
        ordersOfBot = list(self.database.GetOrdersOfBot(pair=pair))

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

    def locked_in_trades(self, quote:str):
        return sum([float(dict(bot)['quote_lockedintrade']) for bot in self.database.GetAllBots() if dict(bot)['status']=='Looking to exit' if dict(bot)['quote']==quote])
