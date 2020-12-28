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

        # Retrieve the credentials from the file
        f = open('../credentials.txt', "r")
        contents = []
        if f.mode == 'r':
            contents = f.read().split('\n')

        self.binance_keys = dict(api_key = contents[0], secret_key=contents[1])

    def live_plot_flask(self):

        application = Flask(__name__)

        # # https://docs.python.org/3/library/logging.html#logging-levels
        # logging.basicConfig(level    = logging.DEBUG,
        #                     filename = os.path.basename(__file__) + '.log',
        #                     format   = "{asctime} [{levelname:8}] {process} {thread} {module}: {message}",
        #                     style    = "{")

        # create instance of BinanceWebSocketApiManager for Binance.com
        try:
            binance_websocket_api_manager = BinanceWebSocketApiManager(exchange="binance.com")
        except requests.exceptions.ConnectionError:
            print("No internet connection?")
            sys.exit(1)

        # create streams
        duration_between_refreshes = 100    # ms
        # Top <levels> bids and asks, pushed every second. Valid <levels> are 5, 10, or 20.
        binance_websocket_api_manager.create_stream(channels     = f"depth20@{duration_between_refreshes}ms",
                                                    markets      = self.base + self.quote,
                                                    stream_label = 'depth20',
                                                    output       = "UnicornFy",
                                                    api_key      = self.binance_keys['api_key'],
                                                    api_secret   = self.binance_keys['secret_key']
                                                    )
        # # Pushes any update to the best bid or ask's price or quantity in real-time for a specified symbol.
        # binance_websocket_api_manager.create_stream(channels     = "bookTicker",
        #                                             markets      = self.base + self.quote,
        #                                             stream_label = 'bookTicker',
        #                                             output       = "UnicornFy",
        #                                             api_key      = self.binance_keys['api_key'],
        #                                             api_secret   = self.binance_keys['secret_key']
        #                                             )

        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(14,12))
        # First plot
        ax4 = ax3.twinx()
        # ax2etdemi = ax1.twinx()

        dates      = []
        imbalances = []
        weighted_bid_volumes = []
        weighted_ask_volumes = []
        screen = pf.screen(canvas = np.zeros((800, 1000), dtype=np.uint8),
                           title  = 'Order Book Live')
        x_max_range = 100

        while True:

            if binance_websocket_api_manager.is_manager_stopping():
                exit(0)

            oldest_stream_data_from_stream_buffer = binance_websocket_api_manager.pop_stream_data_from_stream_buffer()

            if oldest_stream_data_from_stream_buffer is False:
                time.sleep(duration_between_refreshes/1000)

            else:
                if oldest_stream_data_from_stream_buffer is not None:
                    try:
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
                        ax1.scatter(bids_prices, bids_quantities, color='blue',  marker='x', label='bids')       # Blue = red, red=blue (wtf?)
                        ax1.scatter(asks_prices, asks_quantities, color='green', marker='x', label='asks')
                        ax1.axvline(x=midprice, color='black', linestyle='--', label='midprice')
                        ax1.set_xlim(midprice-max(max(bids_prices)-min(bids_prices), max(asks_prices)-min(asks_prices)),
                                     midprice+max(max(bids_prices)-min(bids_prices), max(asks_prices)-min(asks_prices)))

                        ax1.set_title(f'{self.base.upper()}{self.quote.upper()} Order Book')
                        ax1.set_xlabel(f'Price of {self.base} in {self.quote}')
                        ax1.set_ylabel('Base quantity')
                        ax1.legend(loc="upper left")

                        # # Second plot : imbalance time series _______________________________________________________________
                        # dates.append(datetime.fromtimestamp(oldest_stream_data_from_stream_buffer['last_update_id']/1000))
                        dates.append(datetime.fromtimestamp(int(str(oldest_stream_data_from_stream_buffer['last_update_id']))))
                        imbalances.append(imbalance)

                        # Store the best bid and ask data
                        weighted_bid_volumes.append(weighted_bid_volume)
                        weighted_ask_volumes.append(weighted_ask_volume)

                        if len(dates) >= x_max_range:
                            ax3.set_xlim(dates[-x_max_range], dates[-1])
                            # limit the size of the lists so that we only store the displayed data
                            dates = dates[-x_max_range:]
                            imbalances = imbalances[-x_max_range:]
                            weighted_bid_volumes = weighted_bid_volumes[-x_max_range:]
                            weighted_ask_volumes = weighted_ask_volumes[-x_max_range:]
                            # Display dotted lines corresponding to the last max and min values of the imbalance
                            ax3.axhline(y=max(imbalances[-x_max_range:]), c='black', linestyle=':')
                            ax3.axhline(y=min(imbalances[-x_max_range:]), c='black', linestyle=':')
                        ax3.set_ylim(-1.5, 1.5)
                        ax3.plot(dates, imbalances,           c='black', label=f"Imbalance value")
                        ax4.plot(dates, weighted_bid_volumes, c='blue',  label=f"best_bid_price", alpha=0.5)
                        ax4.plot(dates, weighted_ask_volumes, c='green', label=f"best_ask_price", alpha=0.5)
                        # ax4.set_ylim(midprice*0.9999, midprice*1.0001)
                        ax3.set_title(f'Imbalance value')
                        ax3.set_xlabel('Date')
                        ax3.set_ylabel(f'Imbalance value')
                        ax4.set_ylabel(f'Weighted quantities')
                        ax3.legend(loc="upper left")
                        ax3.tick_params(axis='y', color='red')

                        # # Second plot : imbalance time series _______________________________________________________________


                        # If we haven't already shown or saved the plot, then we need to draw the figure first...
                        fig.canvas.draw()
                        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                        # Trying to resize the window, unsuccessfully atm.
                        # figure_dimensions = fig.canvas.get_width_height()[::-1]
                        # print(figure_dimensions)
                        # image = image.reshape(tuple([int(figure_dimensions[0]*0.7), int(figure_dimensions[1]*0.7), 3]))
                        screen.update(image)
                        # screen.update(fig)

                    except Exception:
                        pass



if __name__ == '__main__':

    exchange_ = Binance(filename='../credentials.txt')
    quote_    = 'btc'
    base_     = 'eth'

    order_book = OrderBookAnalysis(exchange=exchange_, quote=quote_, base=base_)

    # order_book.run()
    order_book.live_plot_flask()