from Exchange     import Binance
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime
from operator import itemgetter
from decimal  import Decimal, getcontext
import numpy as np
import pandas as pd
from time import sleep
from tqdm import tqdm
import dash
import gevent
from flask_socketio import SocketIO
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import matplotlib.pyplot as plt
# from matplotlib import dates
from unicorn_binance_websocket_api.unicorn_binance_websocket_api_manager import BinanceWebSocketApiManager
import logging
import os
import requests
import sys
import json
import random
import time
from datetime import datetime, timedelta
import threading
from unicorn_fy.unicorn_fy import UnicornFy
from flask import Flask, Response, render_template
from collections import deque
import matplotlib.animation as animation
from matplotlib.ticker import FuncFormatter
from matplotlib.animation import FuncAnimation
import psutil
import pyformulas as pf
from bokeh.plotting import figure, curdoc


class OrderBookAnalysis:

    def __init__(self, exchange, quote, base):
        self.exchange = exchange
        self.quote = quote
        self.base = base


    def live_plot_flask(self):

        application = Flask(__name__)
        # random.seed()               # Initialize the random number generator

        # https://docs.python.org/3/library/logging.html#logging-levels
        logging.basicConfig(level    = logging.DEBUG,
                            filename = os.path.basename(__file__) + '.log',
                            format   = "{asctime} [{levelname:8}] {process} {thread} {module}: {message}",
                            style    = "{")

        # create instance of BinanceWebSocketApiManager for Binance.com
        try:
            binance_websocket_api_manager = BinanceWebSocketApiManager(exchange="binance.com")
        except requests.exceptions.ConnectionError:
            print("No internet connection?")
            sys.exit(1)

        # set api key and secret for userData stream
        binance_websocket_api_manager.set_private_api_config(self.exchange.binance_keys['api_key'],
                                                             self.exchange.binance_keys['secret_key'])

        # create streams
        # Top <levels> bids and asks, pushed every second. Valid <levels> are 5, 10, or 20.
        binance_websocket_api_manager.create_stream(channels     = "depth20@100ms",
                                                    markets      = self.base + self.quote,
                                                    stream_label = 'depth20',
                                                    output       = "UnicornFy",
                                                    # output       = "raw_data",
                                                    )
        # # Pushes any update to the best bid or ask's price or quantity in real-time for a specified symbol.
        # binance_websocket_api_manager.create_stream(channels     = "bookTicker",
        #                                             markets      = self.base + self.quote,
        #                                             stream_label = 'bookTicker',
        #                                             output       = "UnicornFy",
        #                                             )

        # fig = plt.figure()
        # ax1 = fig.add_subplot(111)
        # ax2 = ax1.twiny()


        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(14,12))
        # First plot
        ax4 = ax3.twinx()
        # ax2etdemi = ax1.twinx()

        dates      = []
        imbalances = []
        weighted_bid_volumes = []
        weighted_ask_volumes = []

        screen = pf.screen(title='Order Book Live')
        wait_secs   = 0.05
        x_max_range = 100

        while True:
            if binance_websocket_api_manager.is_manager_stopping():
                exit(0)
            oldest_stream_data_from_stream_buffer = binance_websocket_api_manager.pop_stream_data_from_stream_buffer()
            if oldest_stream_data_from_stream_buffer is False:
                time.sleep(wait_secs)
            else:
                if oldest_stream_data_from_stream_buffer is not None:
                    try:
                        # print(oldest_stream_data_from_stream_buffer)
                        bids = oldest_stream_data_from_stream_buffer['bids']
                        asks = oldest_stream_data_from_stream_buffer['asks']
                        bids_prices     = np.array([float(bid[0]) for bid in bids])
                        bids_quantities = np.array([float(bid[1]) for bid in bids])
                        asks_prices     = np.array([float(ask[0]) for ask in asks])
                        asks_quantities = np.array([float(ask[1]) for ask in asks])
                        midprice = (asks_prices[0]+bids_prices[0])/2

                        weights = np.exp(-0.5*np.arange(len(bids)))
                        weighted_bid_volume = np.multiply(weights, bids_quantities).sum()    # bids are sorted from larger to smaller prices
                        weighted_ask_volume = np.multiply(weights, asks_quantities).sum()    # asks are sorted from smaller to larger prices

                        # https://tspace.library.utoronto.ca/bitstream/1807/70567/3/Rubisov_Anton_201511_MAS_thesis.pdf     (PAGE 6)
                        imbalance = (weighted_bid_volume - weighted_ask_volume) / (weighted_bid_volume + weighted_ask_volume)       # > 0 : imbalance in favor of bid side, < 0 ask side

                        ax1.clear()
                        ax3.clear()
                        ax4.clear()

                        # First plot ____________________________________________________________________________________
                        ax1.scatter(bids_prices, bids_quantities, color='blue', marker='x', label='bids')       # Blue = red, red=blue (wtf?)
                        # ax1.scatter(bids_prices[0], bids_quantities[0], c="red",   marker='x')
                        ax1.scatter(asks_prices, asks_quantities, color='green', marker='x', label='asks')
                        ax1.axvline(x=midprice, color='black', linestyle='--', label='midprice')
                        ax1.set_xlim(midprice-max(max(bids_prices)-min(bids_prices), max(asks_prices)-min(asks_prices)),
                                     midprice+max(max(bids_prices)-min(bids_prices), max(asks_prices)-min(asks_prices)))
                        # bids_prices_strange     = [bid[1] for bid in bids if bid[0] >= midprice]
                        # bids_quantities_strange = [bid[0] for bid in bids if bid[0] >= midprice]
                        # asks_prices_strange     = [ask[0] for ask in asks if ask[0] <= midprice]
                        # asks_quantities_strange = [ask[1] for ask in asks if ask[0] <= midprice]
                        # if len(bids_prices_strange)>0:
                        #     ax1.scatter(bids_prices_strange, bids_quantities_strange, color="red")
                        # ax1.legend(loc="upper left")
                        ax1.set_title(f'{self.base.upper()}{self.quote.upper()} Order Book')
                        ax1.set_xlabel(f'Price of {self.base} in {self.quote}')
                        ax1.set_ylabel('Base quantity')
                        ax1.legend(loc="upper left")
                        # ax1.tick_params(axis='x',  colors='black')
                        # ax2.axvline(x=imbalance, color='red', linestyle='--',  label='imbalance')
                        # ax2.set_xlim(-1.5,1.5)
                        # ax2.set_xlabel(f'Imbalance value')
                        # ax2.tick_params(axis='x',  colors='black')

                        # # Second plot : imbalance time series ___________________________________________________________
                        # dates.append(datetime.fromtimestamp(oldest_stream_data_from_stream_buffer['last_update_id']/1000))
                        dates.append(datetime.fromtimestamp(int(str(oldest_stream_data_from_stream_buffer['last_update_id']))))
                        imbalances.append(imbalance)
                        # Store the best bid and ask data
                        weighted_bid_volumes.append(weighted_bid_volume)
                        weighted_ask_volumes.append(weighted_ask_volume)

                        print(datetime.fromtimestamp(12340000))

                        # target_date_time_ms = oldest_stream_data_from_stream_buffer['last_update_id']
                        # base_datetime = datetime(1970, 1, 1)
                        # delta = timedelta(target_date_time_ms)
                        # target_date = base_datetime + delta
                        # print(target_date)

                        # ms = oldest_stream_data_from_stream_buffer['last_update_id']
                        # print(datetime.utcfromtimestamp(ms).replace(microsecond=ms%1000*1000))

                        # print(bids_prices[0], asks_prices[0], True if bids_prices[0]>asks_prices[0] else '')

                        if len(dates) >= x_max_range:
                            ax3.set_xlim(dates[-x_max_range], dates[-1])
                            # limit the size of the lists so that we only store the displayed data
                            dates = dates[-x_max_range:]
                            imbalances = imbalances[-x_max_range:]
                            weighted_bid_volumes = weighted_bid_volumes[-x_max_range:]
                            weighted_ask_volumes = weighted_ask_volumes[-x_max_range:]
                            ax3.axhline(y=max(imbalances[-x_max_range:]), c='black', linestyle=':')
                            ax3.axhline(y=min(imbalances[-x_max_range:]), c='black', linestyle=':')
                        ax3.set_ylim(-1.5, 1.5)
                        ax3.plot(dates, imbalances,           c='black', label=f"Imbalance value")
                        ax4.plot(dates, weighted_bid_volumes, c='blue',  label=f"best_bid_price")
                        ax4.plot(dates, weighted_ask_volumes, c='green', label=f"best_ask_price")
                        # ax4.set_ylim(midprice*0.9999, midprice*1.0001)
                        ax3.set_title(f'Imbalance value')
                        ax3.set_xlabel('Date')
                        ax3.set_ylabel(f'Imbalance value')
                        ax4.set_ylabel(f'Weighted quantities')
                        ax3.legend(loc="upper left")
                        ax3.tick_params(axis='y', color='red')
                        # ax3.grid(linestyle='--', axis='y')
                        plt.subplots_adjust(hspace=0.3)

                        # plt.legend()
                        # If we haven't already shown or saved the plot, then we need to draw the figure first...
                        fig.canvas.draw()

                        # print(len(dates), dates)


                        # ax1.clear()
                        # ax2.clear()
                        # ax3.clear()
                        # fig.clear()

                        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                        screen.update(image)

                    except Exception:
                        pass





    def stream_binance(self):

        # https://docs.python.org/3/library/logging.html#logging-levels
        logging.basicConfig(level    = logging.DEBUG,
                            filename = os.path.basename(__file__) + '.log',
                            format   = "{asctime} [{levelname:8}] {process} {thread} {module}: {message}",
                            style    = "{")

        # create instance of BinanceWebSocketApiManager for Binance.com
        try:
            binance_websocket_api_manager = BinanceWebSocketApiManager(exchange="binance.com")
        except requests.exceptions.ConnectionError:
            print("No internet connection?")
            sys.exit(1)

        # set api key and secret for userData stream
        binance_websocket_api_manager.set_private_api_config(self.exchange.binance_keys['api_key'],
                                                             self.exchange.binance_keys['secret_key'])

        # create streams
        # Partial Book Depth Stream
        binance_websocket_api_manager.create_stream(channels     = "kline_1m",
                                                    markets      = "ethbtc",
                                                    stream_label = 'kline_1m',
                                                    output       = "UnicornFy",
                                                    )

        while True:
            if binance_websocket_api_manager.is_manager_stopping():
                exit(0)
            oldest_stream_data_from_stream_buffer = binance_websocket_api_manager.pop_stream_data_from_stream_buffer()
            if oldest_stream_data_from_stream_buffer is False:
                time.sleep(0.01)
            else:
                if oldest_stream_data_from_stream_buffer is not None:
                    try:
                        print(type(oldest_stream_data_from_stream_buffer['kline']['open_price']))
                        # for key, value in oldest_stream_data_from_stream_buffer['kline'].items():
                        #     print(key, value)
                        # print('\n')

                    except KeyError:
                        pass



if __name__ == '__main__':

    exchange_ = Binance(filename='credentials.txt')
    quote_ = 'btc'
    base_  = 'eth'

    order_book = OrderBookAnalysis(exchange=exchange_, quote=quote_, base=base_)


    # order_book.run()
    order_book.live_plot_flask()
    # order_book.stream_binance()



    # imbalances_list = []
    # for i in tqdm(range(20)):
    #     sleep(1)
    #
    #     book = exchange_.GetOrderBook(pair='ETHBTC', limit=5000)
    #
    #     if book:
    #
    #         # lastUpdateId = book['lastUpdateId']
    #
    #         # print(datetime.utcfromtimestamp(lastUpdateId/1000).strftime("%H:%M:%S"))
    #         # print(datetime.utcfromtimestamp(int(lastUpdateId)))
    #
    #         # Sort by price
    #         bids = sorted(book['bids'], key=itemgetter(0))
    #         asks = sorted(book['asks'], key=itemgetter(0))
    #
    #         bids_prices     = np.array([float(bid[0]) for bid in bids])
    #         bids_quantities = np.array([float(bid[1]) for bid in bids])
    #         asks_prices     = np.array([float(ask[0]) for ask in asks])
    #         asks_quantities = np.array([float(ask[1]) for ask in asks])
    #
    #         weights = np.exp(-0.5*np.arange(len(bids)))
    #         weighted_bid_volume = np.multiply(weights, bids_quantities[::-1]).sum()   # Reversed
    #         weighted_ask_volume = np.multiply(weights, asks_quantities).sum()
    #
    #         # https://tspace.library.utoronto.ca/bitstream/1807/70567/3/Rubisov_Anton_201511_MAS_thesis.pdf     (PAGE 6)
    #         imbalance = (weighted_bid_volume - weighted_ask_volume) / (weighted_bid_volume + weighted_ask_volume)       # > 0 : imbalance in favor of bid side, < 0 ask side
    #
    #         imbalances_list.append(imbalance)
    #
    #
    #
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x    = np.arange(len(imbalances_list)),
    #                          y    = imbalances_list,
    #                          mode = 'markers',
    #                          name = 'imbalances_list',
    #                          # line = dict(color='rgb(255,255,51)', width=2),
    #                          ))
    #
    # fig.update_layout({
    #     "margin": {"t": 30, "b": 20},
    #     "height": 800,
    #     "xaxis" : {
    #         # "fixedrange"    : True,
    #         "showline"      : True,
    #         "zeroline"      : False,
    #         "showgrid"      : False,
    #         "showticklabels": True,
    #         "rangeslider"   : {"visible": False},
    #         "color"         : "#a3a7b0",
    #     },
    #     "yaxis" : {
    #         "fixedrange"    : True,
    #         "showline"      : False,
    #         "zeroline"      : False,
    #         "showgrid"      : False,
    #         "showticklabels": True,
    #         "ticks"         : "",
    #         "color"         : "#a3a7b0",
    #     },
    #     "yaxis2" : {
    #         "fixedrange"    : True,
    #         "showline"      : False,
    #         "zeroline"      : False,
    #         "showgrid"      : False,
    #         "showticklabels": True,
    #         "ticks"         : "",
    #         # "color"        : "#a3a7b0",
    #         # "range"         : [0, max(quantity) * 10],
    #     },
    #     "legend" : {
    #         "font"          : dict(size=15, color="#a3a7b0"),
    #     },
    #     "plot_bgcolor"  : "#23272c",
    #     "paper_bgcolor" : "#23272c",
    #
    # })
    # fig.show()

    # fig = make_subplots(specs=[[{"secondary_y": True}]])
    # fig.add_trace(go.Scatter(x    = bids_prices,
    #                          y    = bids_quantities,
    #                          mode = 'markers',
    #                          name = 'Bids',
    #                          # line = dict(color='rgb(255,255,51)', width=2),
    #                          ),
    #               secondary_y = False)
    #
    # fig.add_trace(go.Scatter(x    = asks_prices,
    #                          y    = asks_quantities,
    #                          mode = 'markers',
    #                          name = 'Asks',
    #                          # line = dict(color='rgb(255,255,51)', width=2),
    #                          ),
    #               secondary_y = False)
    #
    # fig.update_layout({
    #     "margin": {"t": 30, "b": 20},
    #     "height": 800,
    #     "xaxis" : {
    #         # "fixedrange"    : True,
    #         "showline"      : True,
    #         "zeroline"      : False,
    #         "showgrid"      : False,
    #         "showticklabels": True,
    #         "rangeslider"   : {"visible": False},
    #         "color"         : "#a3a7b0",
    #     },
    #     "yaxis" : {
    #         "fixedrange"    : True,
    #         "showline"      : False,
    #         "zeroline"      : False,
    #         "showgrid"      : False,
    #         "showticklabels": True,
    #         "ticks"         : "",
    #         "color"         : "#a3a7b0",
    #     },
    #     "yaxis2" : {
    #         "fixedrange"    : True,
    #         "showline"      : False,
    #         "zeroline"      : False,
    #         "showgrid"      : False,
    #         "showticklabels": True,
    #         "ticks"         : "",
    #         # "color"        : "#a3a7b0",
    #         # "range"         : [0, max(quantity) * 10],
    #     },
    #     "legend" : {
    #         "font"          : dict(size=15, color="#a3a7b0"),
    #     },
    #     "plot_bgcolor"  : "#23272c",
    #     "paper_bgcolor" : "#23272c",
    #
    # })
    # fig.show()