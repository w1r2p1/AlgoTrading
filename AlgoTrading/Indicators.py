import pandas as pd
import pandas_ta as ta
import math
from decimal import Decimal
import numpy as np
import plotly.graph_objs as go
import time



# Get the best trigger points using a piecewise-linear approximation, ie the zigzag indicator.
def Add_ZigZag_legacy(df, min_perc_change):
    # No package includes the zigzag indicator it is computed in house.
    # https://www.quantopian.com/posts/zigzag-implementation-example
    # The ZigZag is not used as an indicator but a data labelling tool for the strategy optimization

    dfSeries = pd.Series(df['close'], index=df.index)
    curVal = dfSeries[0]
    curPos = dfSeries.index[0]
    curDir = 1

    dfRes = pd.DataFrame(index=dfSeries.index, columns=['Dir', 'Value'])

    for ln in dfSeries.index:
        if (dfSeries[ln]-curVal)/curVal*100*curDir >= 0:
            curVal = dfSeries[ln]
            curPos = ln
        elif abs((dfSeries[ln]-curVal)/curVal*100) >= min_perc_change:
            dfRes.iloc[curPos, dfRes.columns.get_loc('Value')] = curVal
            dfRes.iloc[curPos, dfRes.columns.get_loc('Dir')]   = curDir
            curVal = dfSeries[ln]
            curPos = ln
            curDir = -1*curDir

    # Convert to float
    dfRes[['Value']] = dfRes[['Value']].astype(float)
    # dfRes = dfRes.interpolate(method='linear')          # ‘linear’: Ignore the index and treat the values as equally spaced.

    df['triggers'] = dfRes['Value']
    # Sort the triggers between buys & sells
    df['buys']  = df['triggers'][df['triggers'] < df['close'].shift(1)]
    df['sells'] = df['triggers'][df['triggers'] > df['close'].shift(1)]
    del df['triggers']

    return df


def Add_ZigZag(df, min_perc_change):
    # No package includes the zigzag indicator it is computed in house.
    # https://www.quantopian.com/posts/zigzag-implementation-example
    # The ZigZag is not used as an indicator but a data labelling tool for the strategy optimization

    closes = pd.Series(df['close'], index=df.index)
    close_price = closes[0]
    close_date  = closes.index[0]
    curDir = 1

    dfRes = pd.DataFrame(index=closes.index, columns=['Dir', 'Pivots'])

    for ln in closes.index:
        perc_change = (closes[ln]-close_price)/close_price*100
        if perc_change*curDir >= 0: # Si les deux sont du même signe ?
            close_price = closes[ln]
            close_date = ln
        elif abs(perc_change) >= min_perc_change:
            dfRes.loc[close_date, 'Pivots'] = close_price
            dfRes.loc[close_date, 'Dir']    = curDir

            close_price = closes[ln]
            close_date = ln
            curDir = -1*curDir  # On cherche dans l'autre sens

    # Convert to float
    dfRes[['Pivots']] = dfRes[['Pivots']].astype(float)
    # dfRes = dfRes.interpolate(method='linear')          # ‘linear’: Ignore the index and treat the values as equally spaced.

    df['pivots'] = dfRes['Pivots']
    # Sort the triggers between buys & sells
    df['buys']  = df['pivots'][df['pivots'] < df['close'].shift(1)]
    df['sells'] = df['pivots'][df['pivots'] > df['close'].shift(1)]
    del df['pivots']

    return df


def Add_ZigZag_pct_change(df, min_change):
    # No package includes the zigzag indicator it is computed in house.
    # https://www.quantopian.com/posts/zigzag-implementation-example
    # The ZigZag is not used as an indicator but a data labelling tool for the strategy optimization

    dfSeries = pd.Series(df['pct_changes'], index=df.index)
    curVal = dfSeries[0]
    curPos = dfSeries.index[0]
    curDir = 1

    dfRes = pd.DataFrame(index=dfSeries.index, columns=['Dir', 'Value'])

    for ln in dfSeries.index:
        if (dfSeries[ln] - curVal)*curDir >= 0:
            curVal = dfSeries[ln]
            curPos = ln
        else:
            if abs(dfSeries[ln]-curVal) >= min_change:
                dfRes.iloc[curPos, dfRes.columns.get_loc('Value')] = curVal
                dfRes.iloc[curPos, dfRes.columns.get_loc('Dir')]   = curDir
                curVal = dfSeries[ln]
                curPos = ln
                curDir = -1*curDir

    # Convert to float
    dfRes[['Value']] = dfRes[['Value']].astype(float)
    # dfRes = dfRes.interpolate(method='linear')          # ‘linear’: Ignore the index and treat the values as equally spaced.

    df['triggers'] = dfRes['Value']
    # Sort the triggers between buys & sells
    df['buys']  = df['triggers'][df['triggers'] < df['pct_changes'].shift(1)]
    df['sells'] = df['triggers'][df['triggers'] > df['pct_changes'].shift(1)]
    del df['triggers']

    return df

def Add_EMA(df, period):

    if 'log_returns' in df.columns:
        df["EMA_" + str(period)] = ta.ema(df['log_returns'], length=period)
    else :
        df["EMA_" + str(period)] = ta.ema(df['close'], length=period)
    return df

def Add_MACD(df, fast:int, slow:int, signal:int):

    # help(ta.macd)
    macd_name   = "MACD_{fast}_{slow}".format(fast=fast, slow=slow)
    signal_name = "MACD_signal_{signal}".format(signal=signal)

    closes = None
    if 'log_returns' in df.columns:
        closes = df.loc[:,'log_returns']
    else:
        closes = df.loc[:,'close']

    # if not df.__contains__(macd_name) and not df.__contains__(signal_name):
    macd = ta.macd(closes, fast=fast, slow=slow, signal=signal)
    df[macd_name]        = macd['MACD_12_26_9']
    df['MACD_histogram'] = macd['MACDH_12_26_9']
    df[signal_name]      = macd['MACDS_12_26_9']
    return df

def Add_BollingerBands(df, length:int, std:int, mamode:str):

    # help(ta.bbands)
    bbands_lower = "BBL_{length}_{std}".format(length=length, std=std)
    bbands_mid   = "BBM_{length}_{std}".format(length=length, std=std)
    bbands_upper = "BBU_{length}_{std}".format(length=length, std=std)

    closes = None
    if 'log_returns' in df.columns:
        closes = df.loc[:,'log_returns']
    else:
        closes = df.loc[:,'close']

    bbands = ta.bbands(closes, length=length, std=std, mamode=mamode)

    df[bbands_lower] = bbands.iloc[:, 0]
    df[bbands_mid]   = bbands.iloc[:, 1]
    df[bbands_upper] = bbands.iloc[:, 2]

    return df

def Add_DonchianChannels(df, lower_length:int, upper_length:int):

    # help(ta.dc)
    lower = "DCL_{lower_length}_{upper_length}".format(lower_length=lower_length, upper_length=upper_length)
    mid   = "DCM_{lower_length}_{upper_length}".format(lower_length=lower_length, upper_length=upper_length)
    upper = "DCU_{lower_length}_{upper_length}".format(lower_length=lower_length, upper_length=upper_length)

    closes = None
    if 'close_log_returns' in df.columns:
        closes = df.loc[:,'close_log_returns']
    else:
        closes = df.loc[:,'close']

    Donchian_Channels = ta.donchian(closes, lower_length=lower_length, upper_length=upper_length)

    df[lower] = Donchian_Channels.iloc[:, 0]
    df[mid]   = Donchian_Channels.iloc[:, 1]
    df[upper] = Donchian_Channels.iloc[:, 2]

    return df

def Add_Normalized_Average_True_Range(df, length:int):

    # help(ta.natr)
    df["NATR_{length}".format(length=length)] = ta.natr(high=df.high, low=df.low, close=df.close, length=length)

    return df



class TestsIndicators:

    def __init__(self, quote, pair, timeframe):
        self.df = self.get_df(quote=quote, pair=pair, timeframe=timeframe)
        self.quote = quote
        self.pair  = pair
        self.timeframe = timeframe


    def test_ZigZag(self):

        min_perc_change = 20
        Add_ZigZag(self.df, min_perc_change=min_perc_change)

        with pd.option_context('display.max_rows', 10, 'display.max_columns', None):
            print(self.df)

        # Plot the prices and pivot points ___________________________
        fig  = go.Figure()

        # Plot the log returns for this pair
        fig.add_trace(go.Scatter(x    = self.df.index,
                                 y    = self.df['close'],
                                 mode = 'lines',
                                 name = 'Prices',
                                 ))
        fig.update_yaxes(title_text  = "<b>" + self.pair.replace(self.quote, '') + "</b> price in " + self.quote)
        fig.update_layout({'yaxis' : {'zeroline' : True},
                           'title' : f"Time series sampled at {timeframe_} intervals. <br>Labels : change is > {min_perc_change}%.",        # <br> = linebreak in plotly
                           }
                          )

        # Add the buys
        fig.add_trace(go.Scatter(x    = self.df.index,
                                 y    = self.df['buys'],
                                 mode   = "markers",
                                 marker = dict(size   = 10,
                                               color  = 'red',
                                               symbol = 'cross',
                                               ),
                                 name = 'Buy points'))
        # Add the sells
        fig.add_trace(go.Scatter(x    = self.df.index,
                                 y    = self.df['sells'],
                                 mode   = "markers",
                                 marker = dict(size   = 10,
                                               color  = 'green',
                                               symbol = 'cross',
                                               ),
                                 name = 'Sell points'))

        # # Add the triggers in zig zag mode
        # pivots_mask  = np.isfinite(self.df.loc[:, 'pivots'].replace({0:np.nan}).astype(np.double))		# mask the NaNs for the plot, to allow to use plot with nan values
        # fig.add_trace(go.Scatter(x    = self.df.index[pivots_mask],
        #                          y    = self.df['pivots'][pivots_mask],
        #                          mode = 'lines',
        #                          line = dict(color='red'),
        #                          name = 'Pivots',
        #                          ))



        # Layout for the main graph
        fig.update_layout({
            'margin': {'t': 100, 'b': 20},
            'height': 800,
            'hovermode': 'x',
            'legend_orientation':'h',

            'xaxis'  : {
                'showline'      : True,
                'zeroline'      : False,
                'showgrid'      : False,
                'showticklabels': True,
                'rangeslider'   : {'visible': False},
                'showspikes'    : True,
                'spikemode'     : 'across+toaxis',
                'spikesnap'     : 'cursor',
                'spikethickness': 0.5,
                'color'         : '#a3a7b0',
            },
            'yaxis'  : {
                # 'autorange'      : True,
                # 'rangemode'     : 'normal',
                # 'fixedrange'    : False,
                'showline'      : False,
                'showgrid'      : False,
                'showticklabels': True,
                'ticks'         : '',
                'showspikes'    : True,
                'spikemode'     : 'across+toaxis',
                'spikesnap'     : 'cursor',
                'spikethickness': 0.5,
                'spikecolor'    : '#a3a7b8',
                'color'         : '#a3a7b0',
            },
            'yaxis2' : {
                # "fixedrange"    : True,
                'showline'      : False,
                'zeroline'      : False,
                'showgrid'      : False,
                'showticklabels': True,
                'ticks'         : '',
                # 'color'        : "#a3a7b0",
            },
            'legend' : {
                'font'          : dict(size=15, color='#a3a7b0'),
            },
            'plot_bgcolor'  : '#23272c',
            'paper_bgcolor' : '#23272c',
        })

        fig.show()


    @staticmethod
    def get_df(quote:str, pair:str, timeframe:str):
        """ Gets the historic data from the csv file """

        # # Work on historical data from the csv file
        historical_data_file = f'historical_data/{quote}/{timeframe}/{pair}_{timeframe}'
        dataframe_ = pd.read_csv(historical_data_file, sep='\t')

        # Remove duplicated lines in the historical data if present
        dataframe = dataframe_.loc[~dataframe_.index.duplicated(keep='first')]

        del dataframe['buys']
        del dataframe['sells']

        # Make the triggers values binary : -1/1
        # Doc : df.loc[<row selection>, <column selection>]
        # dataframe.loc[dataframe.buys.isna(),  'buys']  = 0
        # dataframe.loc[dataframe.buys != 0,    'buys']  = 1
        # dataframe.loc[dataframe.sells.isna(), 'sells'] = 0
        # dataframe.loc[dataframe.sells != 0,   'sells'] = 1

        if 'index' in dataframe.columns:
            del dataframe['index']

        # Set index
        dataframe.set_index('time', inplace=True)

        # Reformat datetime index, Binance's data is messy
        dataframe.index = pd.to_datetime(dataframe.index, format='%Y-%m-%d %H:%M:%S.%f')

        return dataframe.iloc[1:]


if __name__ == '__main__':

    quote_     = 'BTC'
    pair_      = 'ETHBTC'
    timeframe_ = '2h'

    test_class = TestsIndicators(quote=quote_, pair=pair_, timeframe=timeframe_)
    test_class.test_ZigZag()