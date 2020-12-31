import pandas as pd
import plotly.graph_objs as go


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



class TestsIndicators:

    def __init__(self, quote, pair, timeframe):
        self.df = self.get_df(quote=quote, pair=pair, timeframe=timeframe)
        self.quote = quote
        self.pair  = pair
        self.timeframe = timeframe

    def test_ZigZag(self):

        min_perc_change = 20
        Add_ZigZag(self.df, min_perc_change=min_perc_change)

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
    timeframe_ = '1h'

    test_class = TestsIndicators(quote=quote_, pair=pair_, timeframe=timeframe_)
    test_class.test_ZigZag()