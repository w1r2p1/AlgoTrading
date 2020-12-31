from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
from unicorn_binance_websocket_api.unicorn_binance_websocket_api_manager import BinanceWebSocketApiManager
import requests
import time
import numpy as np
from datetime import datetime
import sys
from Exchange import Binance
# from datetime import timedelta
import datetime
from collections import deque

# Parameters for the plot
duration_between_refreshes = 100         # in milliseconds. 1000ms or 100ms
depth  = 20
maxlen = 100

# current_milli_time = lambda: int(round(time.time() * 1000))

class TimeAxisItem(pg.AxisItem):
    # Class that holds the properties for a datetime axis (for the 2nd and 3rd plots)
    def __init__(self, *args, **kwargs):
        super(TimeAxisItem, self).__init__(*args, **kwargs)

    def tickStrings(self, values, scale, spacing):
        return [self.int2dt(value).strftime("%H:%M:%S.%f") for value in values]

    @staticmethod
    def int2dt(ts, ts_mult=1e6):
        return datetime.datetime.utcfromtimestamp(float(ts)/ts_mult)

class RealTimeOrderBook(QtGui.QMainWindow):

    def __init__(self,
                 parent   = None,
                 exchange = Binance(filename='../credentials.txt'),
                 quote    = 'BTC',
                 base     = 'ETH'):
        super(RealTimeOrderBook, self).__init__(parent=parent)

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

        # Top <levels> bids and asks, pushed every X milliseconds. Valid <levels> are 5, 10, or 20.
        self.binance_websocket_api_manager.create_stream(
                                                         channels     = f"depth{depth}@{duration_between_refreshes}ms",
                                                         # channels     = f"depth@{duration_between_refreshes}ms",
                                                         markets      = self.base + self.quote,
                                                         stream_label = f'depth{depth}',
                                                         output       = "UnicornFy",
                                                         api_key      = self.binance_keys['api_key'],
                                                         api_secret   = self.binance_keys['secret_key'],
                                                         )
        # PqtGraph ______________________________________________________________________________________________________________

        # Define a top-level widget to hold everything
        self.mainbox = QtGui.QWidget()
        self.setCentralWidget(self.mainbox)
        self.setWindowTitle('Live OrderBook')
        self.setGeometry(0, 0, 1000, 800)      # synthax : setGeometry(left anchor, top anchor, width, height)

        lay = QtGui.QVBoxLayout(self.mainbox)
        self.canvas = pg.GraphicsLayoutWidget()
        self.label  = QtGui.QLabel()
        lay.addWidget(self.canvas)
        lay.addWidget(self.label)

        # For the fps display
        self.counter = 0
        self.fps = 0.
        self.lastupdate = time.time()

        # Set a timer to update the pots regularly
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(duration_between_refreshes)
        self.timer.start()

        # Construct the window as empty plots that will be populated later on
        # First plot : orderbook ___________________________________________________________________________________________
        self.first_plot = self.canvas.addPlot(title=f'{self.base.upper()}{self.quote.upper()} Order Book')
        self.first_plot.addLegend()     # Must be called before creating the plot
        self.orderbook_bids = self.first_plot.plot([], pen=None, symbolBrush='g', symbolSize=5, symbolPen=None, name="Bids")
        self.orderbook_asks = self.first_plot.plot([], pen=None, symbolBrush='r', symbolSize=5, symbolPen=None, name="Asks")
        self.midprice_line = pg.InfiniteLine(angle=90)  # Vertical line
        self.first_plot.addItem(self.midprice_line)
        self.canvas.nextRow()

        # Second plot : imbalance evolution _______________________________________________________________________________
        self.second_plot = self.canvas.addPlot(title='Imbalance', axisItems={'bottom': TimeAxisItem(orientation='bottom')})
        self.second_plot.addLegend()
        self.imbalance_plt = self.second_plot.plot([], name="Imbalance")
        self.zeroline_imbalance = pg.InfiniteLine(angle=0)  # Horizontal line
        self.second_plot.addItem(self.zeroline_imbalance)
        self.dates      = deque(maxlen=maxlen)              # limit the size of the lists
        self.imbalances = deque(maxlen=maxlen)
        self.canvas.nextRow()

        # Third plot : midprice evolution _________________________________________________________________________________
        self.third_plot = self.canvas.addPlot(title='Midprice', axisItems={'bottom': TimeAxisItem(orientation='bottom')})
        self.third_plot.addLegend()
        self.midprice_plt = self.third_plot.plot([], name="Midprice")
        self.midprices = deque(maxlen=maxlen)
        # self.midprices = deque([1], maxlen=maxlen)
        self.midprices_pct = np.array([])

        # Updarte the graph every X ms
        self.timer.timeout.connect(self.fetch_new_data)


    def fetch_new_data(self):

        dates      = []
        imbalances = []
        weighted_bid_volumes = []
        weighted_ask_volumes = []

        if self.binance_websocket_api_manager.is_manager_stopping():
            exit(0)

        oldest_stream_data_from_stream_buffer = self.binance_websocket_api_manager.pop_stream_data_from_stream_buffer()

        if oldest_stream_data_from_stream_buffer:
            try:
                bids = oldest_stream_data_from_stream_buffer['bids']
                asks = oldest_stream_data_from_stream_buffer['asks']
                bids_prices     = np.array([float(bid[0]) for bid in bids])
                bids_quantities = np.array([float(bid[1]) for bid in bids])
                asks_prices     = np.array([float(ask[0]) for ask in asks])
                asks_quantities = np.array([float(ask[1]) for ask in asks])
                midprice = (asks_prices[0]+bids_prices[0])/2

                # First plot : live orderbook __________________________________________________________________________________________
                self.orderbook_bids.setData(bids_prices, bids_quantities)
                self.orderbook_asks.setData(asks_prices, asks_quantities)
                self.midprice_line.setValue(midprice)
                max_range = max(max(bids_prices)-min(bids_prices), max(asks_prices)-min(asks_prices))
                # self.first_plot.setXRange(midprice - max_range*1.1, midprice + max_range*1.1, padding=0)
                self.first_plot.setXRange(midprice*0.999, midprice*1.001, padding=0)
                self.first_plot.setYRange(0, 30, padding=0)

                # Second plot : imbalance evolution ____________________________________________________________________________________
                weights = np.exp(-0.5*np.arange(len(bids)))
                weighted_bid_volume = np.multiply(weights, bids_quantities).sum()    # bids are sorted from larger to smaller prices
                weighted_ask_volume = np.multiply(weights, asks_quantities).sum()    # asks are sorted from smaller to larger prices
                # https://tspace.library.utoronto.ca/bitstream/1807/70567/3/Rubisov_Anton_201511_MAS_thesis.pdf     (PAGE 6)
                imbalance = (weighted_bid_volume - weighted_ask_volume) / (weighted_bid_volume + weighted_ask_volume)       # > 0 : imbalance in favor of bid side, < 0 ask side
                # # Store the best bid and ask data
                # weighted_bid_volumes.append(weighted_bid_volume)
                # weighted_ask_volumes.append(weighted_ask_volume)

                self.dates.append(time.time_ns() // 1000000)
                # self.dates.append(oldest_stream_data_from_stream_buffer['last_update_id']/1000)
                # self.dates.append(oldest_stream_data_from_stream_buffer['event_time']/1000)
                self.imbalances.append(imbalance)
                self.imbalance_plt.setData(x=list(self.dates), y=list(self.imbalances))
                self.zeroline_imbalance.setValue(0) # Horizontal zeroline

                # Third plot : midprice evolution _____________________________________________________________________________________
                self.midprices.append(midprice)
                self.midprice_plt.setData(x=list(self.dates), y=list(self.midprices))
                # self.midprices_pct = np.diff(self.midprices) / midprice * 100.
                # self.midprice_plt.setData(x=list(self.dates)[1:], y=list(self.midprices_pct))


                # Compute and display the fps in real time ____________________________________________________________________________
                now = time.time()
                dt  = (now-self.lastupdate)
                if dt <= 0:
                    dt = 0.000000000001
                fps2 = 1.0 / dt
                self.lastupdate = now
                self.fps = self.fps * 0.9 + fps2 * 0.1
                tx = 'Frame Rate: {fps:.1f} FPS'.format(fps=self.fps)
                self.label.setText(tx)
                self.counter += 1

            except Exception:
                pass



if __name__ == "__main__":

    # print(datetime.datetime.utcfromtimestamp(int(1556876873656/1000)))
    # print(datetime.datetime.utcfromtimestamp(123456785))
    # print(datetime.datetime.utcfromtimestamp(1499827319559//1000))
    # print(datetime.datetime.utcfromtimestamp(2046077850//1000))
    # timestamp = 1556876873656
    # mytime = datetime.datetime.fromtimestamp(timestamp/1000).replace(microsecond = (timestamp % 1000) * 1000)
    # print(mytime)
    #
    #
    # milliseconds = 0
    # if len(str(timestamp)) == 13:
    #     milliseconds = int(str(timestamp)[-3:])
    #     timestamp = float(str(timestamp)[0:-3])
    #
    # the_date = datetime.datetime.fromtimestamp(timestamp)
    # the_date += datetime.timedelta(milliseconds=milliseconds)
    # print(the_date)




    app = QtGui.QApplication(sys.argv)
    thisapp = RealTimeOrderBook()
    thisapp.show()
    sys.exit(app.exec_())
