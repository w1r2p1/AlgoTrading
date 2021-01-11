from decimal import Decimal
import datetime
from   datetime import datetime, timedelta
import socket
import struct
import ctypes, sys
import time
import datetime
import win32api
from   Database import BotDatabase


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
        return len([dict(bot)['pair'] for bot in self.database.get_all_bots() if dict(bot)['status'] == 'Looking to exit' if dict(bot)['quote'] == quote])

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
        return len([dict(order)['pair'] for order in list(self.database.get_orders_of_bot(pair))
                    if datetime.utcnow()-timedelta(hours=24) < datetime.strptime(dict(order)['transactTime'], "%Y-%m-%d %H:%M:%S")])

    def pair_average_hold_duration(self, pair:str):
        """ Computes the average hold duration on a quote."""

        hold_durations = []
        ordersOfBot = list(self.database.get_orders_of_bot(pair=pair))

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
        return sum([float(dict(bot)['quote_lockedintrade']) for bot in self.database.get_all_bots() if dict(bot)['status'] == 'Looking to exit' if dict(bot)['quote'] == quote])

    @staticmethod
    def RoundToValidPrice(bot:dict, price:Decimal)->Decimal:
        """ Addresses the issue of PRICE_FILTER """
        # https://github.com/binance-exchange/binance-official-api-docs/blob/master/rest-api.md#price_filter

        """ The price/stopPrice must obey 3 rules :
                price >= minPrice
                price <= maxPrice
                (price-minPrice) % tickSize == 0 
        """

        minPrice = Decimal(bot['minPrice'])
        maxPrice = Decimal(bot['maxPrice'])
        tickSize = Decimal(bot['tickSize'])

        if minPrice <= price <= maxPrice:
            if (price-minPrice) % tickSize == 0:
                # Round down to the nearest tickSize and remove zeros after
                newPrice = price // tickSize * tickSize
                # print("Attention on {pair} : the price {price} is incorrect and has been rounded to {newPrice}".format(pair=pair, price=price, newPrice=newPrice))
                return newPrice
            else:
                return price

        if price < minPrice:
            # print("Attention on {pair} : the price {price} is too low and has been set to {minPrice}".format(pair=pair, price=price, minPrice=minPrice))
            return minPrice

        if price > maxPrice:
            # print("Attention on {pair} : the price {price} is too high and has been set to {maxPrice}".format(pair=pair, price=price, maxPrice=maxPrice))
            return maxPrice

    @staticmethod
    def RoundToValidQuantity(bot:dict, quantity:Decimal)->Decimal:
            """ Addresses the issue of LOT_SIZE """
            # https://github.com/binance-exchange/binance-official-api-docs/blob/master/rest-api.md#lot_size

            """ The price/stopPrice must obey 3 rules :
                    quantity >= minQty
                    quantity <= maxQty
                    (quantity-minQty) % stepSize == 0
            """

            minQty   = Decimal(bot['minQty'])
            maxQty   = Decimal(bot['maxQty'])
            stepSize = Decimal(bot['stepSize'])

            if minQty <= quantity <= maxQty:
                if (quantity-minQty) % stepSize != 0:
                    # Round down to the nearest stepSize and remove zeros after
                    newQuantity = quantity // stepSize * stepSize
                    return newQuantity
                else:
                    return quantity

            if quantity < minQty:
                return minQty

            if quantity > maxQty:
                return maxQty

    @staticmethod
    def gettime_ntp(addr='time.nist.gov'):

        # http://code.activestate.com/recipes/117211-simple-very-sntp-client/
        TIME1970 = 2208988800      # Thanks to F.Lundh
        client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        data = '\x1b' + 47 * '\0'
        data = data.encode()
        try:
            # Timing out the connection after 5 seconds, if no response received
            client.settimeout(5.0)
            client.sendto(data, (addr, 123))
            data, address = client.recvfrom(1024)
            if data:
                epoch_time = struct.unpack('!12I', data)[10]
                epoch_time -= TIME1970
                return epoch_time
        except socket.timeout:
            return None

    def set_ntp_time(self):
        # List of servers in order of attempt of fetching
        server_list = ['ntp.iitb.ac.in', 'time.nist.gov', 'time.windows.com', 'pool.ntp.org']

        # Needs admin privileges to change the clock
        # ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, __file__, None, 1)

        # Iterates over every server in the list until it finds time from any one.
        for server in server_list:
            epoch_time = self.gettime_ntp(server)
            print(datetime.datetime.fromtimestamp(epoch_time))
            if epoch_time is not None:
                # SetSystemTime takes time as argument in UTC time. UTC time is obtained using utcfromtimestamp()
                utcTime = datetime.datetime.utcfromtimestamp(epoch_time)
                win32api.SetSystemTime(utcTime.year, utcTime.month, utcTime.weekday(), utcTime.day, utcTime.hour, utcTime.minute, utcTime.second, 0)
                # Local time is obtained using fromtimestamp()
                localTime = datetime.datetime.fromtimestamp(epoch_time)
                print(f'Time updated to: {localTime.strftime("%Y-%m-%d %H:%M")} from {server}')
                break
            else:
                print(f"Could not find time from {server}.")


if __name__ == "__main__":

    def is_admin():
        try:
            return ctypes.windll.shell32.IsUserAnAdmin()
        except:
            return False
    print(is_admin())

    HelperMethods(database=BotDatabase(name="assets/database_paper.db")).set_ntp_time()