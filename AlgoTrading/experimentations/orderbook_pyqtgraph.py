# from PyQt5 import QtCore, QtWidgets, QtGui
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
from unicorn_binance_websocket_api.unicorn_binance_websocket_api_manager import BinanceWebSocketApiManager
import requests
import time
import numpy as np
from datetime import datetime
import sys
from Exchange import Binance

duration_between_refreshes = 100         # in milliseconds

class MyWidget(QtGui.QMainWindow):

    def __init__(self,
                 parent   = None,
                 exchange = Binance(filename='../credentials.txt'),
                 quote    = 'BTC',
                 base     = 'ETH'):
        super(MyWidget, self).__init__(parent=parent)

        # Websocket ______________________________________________________________________________________________________________
        self.exchange = exchange
        self.quote    = quote
        self.base     = base

        # Retrieve the credentials from the file
        f = open('../credentials.txt', "r")
        contents = []
        if f.mode == 'r':
            contents = f.read().split('\n')
        self.binance_keys = dict(api_key    = contents[0],
                                 secret_key = contents[1])
        try:
            self.binance_websocket_api_manager = BinanceWebSocketApiManager(exchange="binance.com")
        except requests.exceptions.ConnectionError:
            print("No internet connection?")
            sys.exit(1)

        # Top <levels> bids and asks, pushed every second. Valid <levels> are 5, 10, or 20.
        self.binance_websocket_api_manager.create_stream(channels     = f"depth20@{duration_between_refreshes}ms",
                                                         markets      = self.base + self.quote,
                                                         stream_label = 'depth20',
                                                         output       = "UnicornFy",
                                                         api_key      = self.binance_keys['api_key'],
                                                         api_secret   = self.binance_keys['secret_key']
                                                         )
        # PqtGraph ______________________________________________________________________________________________________________

        # Define a top-level widget to hold everything
        self.mainbox = QtGui.QWidget()
        self.setCentralWidget(self.mainbox)

        lay = QtGui.QVBoxLayout(self.mainbox)
        self.canvas = pg.GraphicsLayoutWidget()
        self.label  = QtGui.QLabel()
        lay.addWidget(self.canvas)
        lay.addWidget(self.label)

        # self.mainbox = QtGui.QWidget()
        # self.setCentralWidget(self.mainbox)
        # self.mainbox.setLayout(QtGui.QVBoxLayout())
        #
        # self.canvas = pg.GraphicsLayoutWidget()
        # self.mainbox.layout().addWidget(self.canvas)
        #
        # self.label = QtGui.QLabel()
        # self.mainbox.layout().addWidget(self.label)


        # For the fps display
        self.counter = 0
        self.fps = 0.
        self.lastupdate = time.time()

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(duration_between_refreshes)
        self.timer.start()

        # Construct the window as empty plots that will be populated later on
        # First plot : orderbook
        self.first_plot = self.canvas.addPlot(title=f'{self.base.upper()}{self.quote.upper()} Order Book')
        self.orderbook_bids = self.first_plot.plot([], pen=None, symbolBrush='g', symbolSize=5, symbolPen=None)
        self.orderbook_asks = self.first_plot.plot([], pen=None, symbolBrush='r', symbolSize=5, symbolPen=None)
        self.midprice_line = pg.InfiniteLine()
        self.first_plot.addItem(self.midprice_line)
        self.canvas.nextRow()
        # Second plot : imbalance
        second_plot = self.canvas.addPlot(title='Imbalance')
        self.imbalance = second_plot.plot([], pen=None, symbolBrush=(255,0,0), symbolSize=5, symbolPen=None)

        # Updarte the graph every X ms
        self.timer.timeout.connect(self.fetch_new_data)


    def fetch_new_data(self):

        dates      = []
        imbalances = []
        weighted_bid_volumes = []
        weighted_ask_volumes = []
        x_max_range = 100

        if self.binance_websocket_api_manager.is_manager_stopping():
            exit(0)

        oldest_stream_data_from_stream_buffer = self.binance_websocket_api_manager.pop_stream_data_from_stream_buffer()


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

                # First plot : live orderbook _____________________________________________________________________________
                self.orderbook_bids.setData(bids_prices, bids_quantities)
                self.orderbook_asks.setData(asks_prices, asks_quantities)
                self.midprice_line.setValue(midprice)
                max_range = max(max(bids_prices)-min(bids_prices), max(asks_prices)-min(asks_prices))
                self.first_plot.setXRange(midprice - max_range*1.1,
                                          midprice + max_range*1.1,
                                          padding=0)

                # # Second plot : imbalance time series ___________________________________________________________________
                # dates.append(datetime.fromtimestamp(oldest_stream_data_from_stream_buffer['last_update_id']/1000))
                dates.append(datetime.fromtimestamp(int(str(oldest_stream_data_from_stream_buffer['last_update_id']))))
                imbalances.append(imbalance)

                # Store the best bid and ask data
                weighted_bid_volumes.append(weighted_bid_volume)
                weighted_ask_volumes.append(weighted_ask_volume)

                if len(dates) >= x_max_range:
                    # ax3.set_xlim(dates[-x_max_range], dates[-1])
                    # self.imbalance.setRange(xRange=[5,20])
                    # limit the size of the lists so that we only store the displayed data
                    dates = dates[-x_max_range:]
                    imbalances = imbalances[-x_max_range:]
                    weighted_bid_volumes = weighted_bid_volumes[-x_max_range:]
                    weighted_ask_volumes = weighted_ask_volumes[-x_max_range:]
                    # Display dotted lines corresponding to the last max and min values of the imbalance
                    # ax3.axhline(y=max(imbalances[-x_max_range:]), c='black', linestyle=':')
                    # ax3.axhline(y=min(imbalances[-x_max_range:]), c='black', linestyle=':')
                # ax3.set_ylim(-1.5, 1.5)
                # ax3.plot(dates, imbalances,           c='black', label=f"Imbalance value")
                # ax4.plot(dates, weighted_bid_volumes, c='blue',  label=f"best_bid_price", alpha=0.5)
                # ax4.plot(dates, weighted_ask_volumes, c='green', label=f"best_ask_price", alpha=0.5)
                # # ax4.set_ylim(midprice*0.9999, midprice*1.0001)
                # ax3.set_title(f'Imbalance value')
                # ax3.set_xlabel('Date')
                # ax3.set_ylabel(f'Imbalance value')
                # ax4.set_ylabel(f'Weighted quantities')
                # ax3.legend(loc="upper left")
                # ax3.tick_params(axis='y', color='red')
                self.imbalance.setData(dates, imbalances)


            # Compute and display the fps in real time
                now = time.time()
                dt  = (now-self.lastupdate)
                if dt <= 0:
                    dt = 0.000000000001
                fps2 = 1.0 / dt
                self.lastupdate = now
                self.fps = self.fps * 0.9 + fps2 * 0.1
                tx = 'Mean Frame Rate: {fps:.1f} FPS'.format(fps=self.fps)
                self.label.setText(tx)
                self.counter += 1

            except Exception:
                pass



def main():
    app = QtGui.QApplication(sys.argv)
    thisapp = MyWidget()
    thisapp.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()