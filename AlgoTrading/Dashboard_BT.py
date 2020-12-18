from Exchange import Binance
from Database import BotDatabase
from HistoricalPrices import indicators_list
from BackTesting import BackTesting
import Settings
# from Strategies import *
import Strategies

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import pickle



"""________________________________________________________________________________________"""

exchange = Binance(filename='credentials.txt')
database = BotDatabase("database.db")

backtest = BackTesting(exchange=exchange, strategy=Settings.parameters['strategy'])

# Work on the quotes that have a directory in 'Historical_data/'
ExistingQuoteassets = next(os.walk('Historical_data/'))[1]                                      # ['ETH', 'BTC']
timeframe = next(os.walk('Historical_data/' + ExistingQuoteassets[0]))[2][0].split('_')[1]      # '1h'

# Work on the pairs that have a file in the sub-directory, ie pairs that we have data for.
pairs = dict()                                                                                  # {'ETH': ['QTUMETH', 'EOSETH',..], 'BTC':[]}
for quote in ExistingQuoteassets:
    # pairs[quote] = [f.replace('_' + timeframe, '') for f in listdir('Historical_data/' + quote + '/') if isfile(join('Historical_data/' + quote + '/', f))]
    pairs[quote] = sorted(list(set([f.strip('_ret').strip('_log').strip('_pct').strip('_' + timeframe) for f in listdir('Historical_data/' + quote + '/') if isfile(join('Historical_data/' + quote + '/', f))])))


"""________________________________________________________________________________________"""

# ____________________________ PAIR ____________________________
def plotPairHistory(quoteasset:str, pair:str, ClosesOrReturns:str, nb_trigger_points:str, indicators:list, clickForBacktest:int):
    """ Plots the candlestick chart with overlays for a pair, over (part of) the backtesting duration. """

    # Create figure with secondary y-axis for the volume
    fig = make_subplots(shared_xaxes=True, specs=[[{"secondary_y": True}]])

    # Get the dataframe from the csv file
    filename = 'historical_data/{quote}/{pair}_{timeframe}{closesorreturns}'.format(quote='BTC', pair='ADABTC', timeframe='1h', closesorreturns=ClosesOrReturns)
    df = pd.read_csv(filename, sep='\t')
    historical_data_fileTEMP = 'historical_data/{quote}/{pair}_{timeframe}_log'.format(quote='BTC', pair='ADABTC', timeframe='1h')
    df_TEMP = pd.read_csv(historical_data_fileTEMP, sep='\t')    # [-1500:]

    # Save the total number of triggers in the original dataframe
    total_triggers = df['buys'].count() + df['sells'].count()
    total_buys     = df['buys'].count()
    total_sells    = df['sells'].count()

    # Take only the relevant part of the dataframe
    if nb_trigger_points != 'all':
        buys_list = df['buys'].to_list()
        first_trigger_to_plot = [i for i, x in enumerate(buys_list) if pd.notna(df['buys'][i])][-int(nb_trigger_points)]             # list comprehension of indices of 'buy', we take the last X ones.
        df = df.iloc[first_trigger_to_plot:]
        df.reset_index(inplace=True)
        df_TEMP = df_TEMP.iloc[first_trigger_to_plot:]
        df_TEMP.reset_index(inplace=True)

    # Plot
    if ClosesOrReturns == '':
        if 'candles' in indicators:
            # plot candlestick chart for this pair
            fig.add_trace(go.Candlestick(x 	   = df['time'],
                                         open  = df['open'],
                                         close = df['close'],
                                         high  = df['high'],
                                         low   = df['low'],
                                         name  = "Candlesticks",
                                         opacity = 0.5),
                          secondary_y = False)
            # Set line and fill colors of candles
            cs = fig.data[0]
            cs.increasing.fillcolor  = '#3D9970'    # '#008000'
            cs.increasing.line.color = '#3D9970'    # '#008000'
            cs.decreasing.fillcolor  = '#FF4136'    # '#800000'
            cs.decreasing.line.color = '#FF4136'    # '#800000'

        if 'prices' in indicators:
            # plot the close prices for this pair
            fig.add_trace(go.Scatter(x    = df['time'],
                                     y    = df['close'],
                                     mode = 'lines',
                                     name = 'Close prices',
                                     line = dict(color='rgb(255,255,51)', width=1.5)),
                          secondary_y = False)

        fig.update_yaxes(title_text  = "<b>" + pair.replace(quoteasset, '') + "</b> price in <b>" + quoteasset + " </b>",
                         secondary_y = False)

    elif ClosesOrReturns == '_log':
        # plot the log returns for this pair
        fig.add_trace(go.Scatter(x    = df['time'],
                                 y    = df['close_log_returns'],
                                 mode = 'lines',
                                 name = 'Log returns'),
                      secondary_y = False)
        fig.update_yaxes(title_text  = "<b>" + pair.replace(quoteasset, '') + "</b> log returns",
                         secondary_y = False)
        fig.update_layout({"yaxis" : {"zeroline" : True}})

    # Add the volume
    fig.add_trace(go.Bar(x       = df['time'],
                         y       = df['volume'],
                         name    = "Volume",
                         marker  = dict(color='#a3a7b0')),
                  secondary_y = True)

    # Display the buy & sell trigger points
    if 'triggers' in indicators:
        if df['buys'] is not None:
            fig.add_trace(go.Scatter(x 		= df['time'],
                                     y 		= df['buys'],
                                     name 	= "Buy triggers",
                                     mode   = "markers",
                                     marker = dict(size  = 10,
                                                   color = 'darkred',
                                                   line  = dict(width=1.5, color='darkred'))))
        # if df['sells'] is not None:
        #     fig.add_trace(go.Scatter(x 	    = df['time'],
        #                              y 	    = df['sells'],
        #                              name 	= "Sell triggers",
        #                              mode 	= "markers",
        #                              marker = dict(size  = 10,
        #                                            color = 'darkgreen',
        #                                            line  = dict(width=1.5, color='darkgreen'))))

    # Run a backtest and display results when clicking the button
    nb_backtest_orders = 0
    nb_backtest_buys   = 0
    nb_backtest_sells  = 0
    Backtest_Profit    = 0
    Backtest_Fees      = 0
    Backtest_ProfitMinusFees = 0
    starting_balance   = 1
    backtest_graph     = None
    if clickForBacktest:

        # df_bt = backtest.Backtest(quote=quoteasset, pair=pair, starting_balance=starting_balance)

        # load the model from disk
        model_file_buys    = '{pair}_{timeframe}_model_{side}.sav'.format(pair='ADABTC', timeframe='1h', side='buys')
        model_file_sells   = '{pair}_{timeframe}_model_{side}.sav'.format(pair='ADABTC', timeframe='1h', side='sells')
        loaded_model_buys  = pickle.load(open(model_file_buys, 'rb'))
        loaded_model_sells = pickle.load(open(model_file_sells, 'rb'))

        df['backtest_buys']  = loaded_model_buys.predict(X)
        df['backtest_sells'] = loaded_model_sells.predict(X)



        fig.add_trace(go.Scatter(
                                 # x 		= [item[0] for item in pair_results['buy_times']],
                                 # y 		= [item[1] for item in pair_results['buy_times']],
                                 # Use this when backtesting with np.where
                                 x 		= df['time'],
                                 y 		= df['backtest_buys'],
                                 name 	= "Buys of backtest",
                                 mode   = "markers",
                                 marker = dict(size   = 10,
                                               color  = 'red',
                                               symbol = 'cross',
                                               line   = dict(width=0.5, color='red'))))

        fig.add_trace(go.Scatter(
                                 # x 		= [item[0] for item in pair_results['sell_times']],
                                 # y 		= [item[1] for item in pair_results['sell_times']],
                                 # Use this when backtesting with np.where
                                 x 		= df['time'],
                                 y 		= df['backtest_sells'],
                                 name 	= "Sells of backtest",
                                 mode 	= "markers",
                                 marker = dict(size   = 10,
                                               color  = 'green',
                                               symbol = 'cross',
                                               line   = dict(width=0.5, color='green'))))
        #
        # # Number of orders
        # nb_backtest_buys   = len(df['backtest_buys'].dropna())
        # nb_backtest_sells  = len(df['backtest_sells'].dropna())
        # nb_backtest_orders = nb_backtest_buys+nb_backtest_sells
        #
        # # Stats
        # Backtest_Profit = [x for x in df['backtest_balance'].to_list() if str(x) != 'nan'][-1] - starting_balance
        # Backtest_Fees   = sum([x for x in df['backtest_fees'].to_list() if str(x) != 'nan'])
        # Backtest_ProfitMinusFees = Backtest_Profit - Backtest_Fees
        #
        # # Plot the backtest graph
        # fig_backtest = go.Figure()
        # fig_backtest.add_trace(go.Scatter(
        #                                     x 	   = df['time'],
        #                                     y 	   = df['backtest_balance'],
        #                                     mode   = 'lines',
        #                                     name   = 'Backtest balance',
        #                                     line   = dict(color='rgb(255,255,51)', width=1.5),
        #                                     connectgaps = True,     # override default to connect the gaps
        #                                     ))
        # fig_backtest.update_layout({
        #     'margin' : {'t': 30, 'b': 20},
        #     'height' : 200,
        #     'title'  : dict(text='Backtest balance',
        #                     font=dict(color='#a3a7b0'),
        #                     y=0.9,
        #                     x=0.5,
        #                     xanchor= 'center',
        #                     yanchor= 'top'),
        #
        #     'xaxis'  : {
        #         'range'         : [df['time'].to_list()[0], df['time'].to_list()[-1]],
        #         'showline'      : True,
        #         'zeroline'      : False,
        #         'showgrid'      : False,
        #         'showticklabels': True,
        #         'rangeslider'   : {'visible': False},
        #         'color'         : '#a3a7b0',
        #     },
        #     'yaxis'  : {
        #         'fixedrange'    : False,
        #         'showline'      : False,
        #         'zeroline'      : True,
        #         'showgrid'      : True,
        #         'showticklabels': True,
        #         'ticks'         : '',
        #         'color'         : '#a3a7b0',
        #     },
        #     'legend' : {
        #         'font'          : dict(size=12, color='#a3a7b0'),
        #     },
        #     'plot_bgcolor'  : '#23272c',
        #     'paper_bgcolor' : '#23272c',
        #
        # })
        # backtest_graph = dcc.Graph(id='backtest_graph', figure=fig_backtest, config={"doubleClick": "reset"})


    # ___________________________ LAYOUT ___________________________________________

    # Layout for the main graph
    fig.update_layout({
        'margin': {'t': 30, 'b': 20},
        'height': 600,
        'hovermode': 'x',
        'legend_orientation':'h',

        'xaxis'  : {
            'range'         : [df['time'].to_list()[0], df['time'].to_list()[-1]],
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
            'range'         : [0, df['volume'].max() * 10],
        },
        'legend' : {
            'font'          : dict(size=15, color='#a3a7b0'),
        },
        'plot_bgcolor'  : '#23272c',
        'paper_bgcolor' : '#23272c',

    })

    # Set secondary y-axe's title
    fig.update_yaxes(title_text="<b>Volume</b>", secondary_y=True)

    # Area below the dropdown menus
    h = html.Div(
                [
                    # dbc.Row([dbc.Col(html.P(".", style={'text-align': 'center'}), width=12)]),    # Simple point pour rajouter une ligne "vide"

                    # Nom de la pair
                    dbc.Row([
                        dbc.Col(width=1),
                        dbc.Col(dbc.Table(html.Tbody(html.Tr(html.Td(pair, style={'text-align': 'center'}))),
                                          bordered   = False,
                                          borderless = True,
                                          dark       = True,
                                          striped    = True,
                                          ),
                                width=10),
                        dbc.Col(width=1),
                        ]),

                    # Tableau des stats
                    dbc.Row([
                        dbc.Col(width=1),
                        # Stats, tableau de gauche
                        dbc.Col(dbc.Table(
                                    html.Tbody(
                                        [
                                            html.Tr([html.Td("Timeframe"),              html.Td(timeframe,      style={'color':'#a3a7b0'})]),

                                            html.Tr([html.Td("Triggers"),               html.Td(total_triggers, style={'color':'#a3a7b0'})]),

                                            html.Tr([html.Td("Buy triggers"),           html.Td(total_buys,     style={'color':'#a3a7b0'})]),

                                            html.Tr([html.Td("Sell triggers"),          html.Td(total_sells,    style={'color':'#a3a7b0'})]),

                                        ]
                                    ),
                                bordered   = False,
                                borderless = True,
                                dark       = True,
                                striped    = True,
                                ),
                            width=5),

                        # Stats, tableau de droite
                        dbc.Col(dbc.Table(html.Tbody(
                                        [
                                            html.Tr([html.Td("Detected buys",style={'whiteSpace': 'pre-line', 'vertical-align': 'middle'}),    html.Td(0, style={'color':'#a3a7b0', 'whiteSpace': 'pre-line'})]),
                                        ]
                                    ),
                                        bordered   = False,
                                        borderless = True,
                                        dark       = True,
                                        striped    = True,
                                    ),
                                    width=5),

                        dbc.Col(width=1),
                    ]),

                    # Graph of the pair
                    dbc.Row(dbc.Col(dcc.Graph(id     = 'pair_graph',
                                              figure = fig,
                                              config = {"doubleClick": "reset"}))),

                    # Tableau du backtest
                    dbc.Row([
                        dbc.Col(width=1),
                        dbc.Col(dbc.Table(html.Tbody(html.Tr(html.Td('Backest Results', style={'text-align': 'center'}))),
                                          bordered   = False,
                                          borderless = True,
                                          dark       = True,
                                          striped    = True,
                                          ),
                                width=10),
                        dbc.Col(width=1),
                    ]),
                    dbc.Row([
                        dbc.Col(width=1),
                        # Backtest, tableau de gauche
                        dbc.Col(dbc.Table(
                            html.Tbody(
                                [
                                    html.Tr([html.Td("Orders"),         html.Td(nb_backtest_orders, style={'color':'#a3a7b0'})]),

                                    html.Tr([html.Td("Buys"),           html.Td(nb_backtest_buys,   style={'color':'#a3a7b0'})]),

                                    html.Tr([html.Td("Sells"),          html.Td(nb_backtest_sells,  style={'color':'#a3a7b0'})]),

                                ]
                            ),
                            bordered   = False,
                            borderless = True,
                            dark       = True,
                            striped    = True,
                        ),
                            width=5),

                        # Backtest, tableau de droite
                        dbc.Col(dbc.Table(html.Tbody(
                            [
                                html.Tr([html.Td("Started with"),       html.Td(str(starting_balance) + " " + quoteasset, style={'color':'#a3a7b0'})]),

                                html.Tr([html.Td("Profit"),             html.Td(str(round(Backtest_Profit, 8)) + " " + quoteasset + " (" + str(round((Backtest_Profit/starting_balance)*100, 2)) + "%)", style={'color':'#a3a7b0'})]),

                                html.Tr([html.Td("Fees"),               html.Td(str(round(Backtest_Fees, 8)) + " " + quoteasset, style={'color':'#a3a7b0'})]),

                                html.Tr([html.Td("Profit - fees"),      html.Td(str(round(Backtest_ProfitMinusFees, 8)) + " " + quoteasset + " (" + str(round((Backtest_ProfitMinusFees/starting_balance)*100, 2)) + "%)", style={'color':'#a3a7b0'})]),

                            ]
                        ),
                            bordered   = False,
                            borderless = True,
                            dark       = True,
                            striped    = True,
                        ),
                            width=5),

                        dbc.Col(width=1),
                    ]),


                    # Backtest graph
                    dbc.Row(dbc.Col([backtest_graph])),


                ]
                )
    return h


# ___________________________ APP LAYOUT ___________________________________________
app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                meta_tags=[{"name": "viewport", "content": "width=device-width"}])

app.layout = html.Div(
    [
        html.Div(
            id        = "header",
            children  = [html.Div([html.H2("BackTesting")], className="eight columns"),
                         html.Div([html.Img(id="logo", src=app.get_asset_url("dash-logo-new.png"))])],
            className = "row"),

        html.Hr(),

        dbc.Row(dbc.Col(html.H3("Specific pair"))),

        # 1st row
        dbc.Row([dbc.Col(width=1),
                 dbc.Col(dcc.Dropdown(id          = 'Dropdown_quote',
                                      options     = [{'label': quote, 'value': quote} for quote in pairs.keys()],
                                      value       = list(pairs.keys())[0],
                                      clearable   = False,
                                      placeholder = "Select a quote"),
                         width=3),
                 dbc.Col(width=0.5),
                 dbc.Col(dcc.Dropdown(id          = 'Dropdown_ClosesOrReturns',
                                      options     = [{'label': 'Prices',      'value': ''},
                                                     {'label': 'Log returns', 'value': '_log'}],
                                      value       = '_log',
                                      placeholder = "Close prices or log returns ?",
                                      clearable   = False),
                         width=3),
                 dbc.Col(width=5),
                 ]),

        # 2nd row
        dbc.Row([dbc.Col(width=1),
                 dbc.Col(dcc.Dropdown(id          = 'Dropdown_pair',
                                      placeholder = "Select a pair"),
                         width=3),
                 dbc.Col(width=0.5),
                 dbc.Col(dcc.Dropdown(id        = 'Dropdown_duration',
                                      options   = [{'label': 'Last buy trigger point',      'value': '1'},
                                                   {'label': 'Last 5 buy trigger points',   'value': '5'},
                                                   {'label': 'Last 10 buy trigger points',  'value': '10'},
                                                   {'label': 'Last 25 buy trigger points',  'value': '25'},
                                                   {'label': 'Since beginning',             'value': 'all'}],
                                      value     = '25',
                                      clearable = False),
                         width=3),
                 dbc.Col(dcc.Dropdown(id        = 'Dropdown_indicators',
                                      value     = ['prices', 'triggers'],
                                      multi     = True,
                                      clearable = False,
                                      style     = dict()),
                         width=3),
                 dbc.Col(width=1)]),


        # Button to run a backtest on this pair
        dbc.Row([dbc.Col(width=1),
                 # Run a backtest
                 dbc.Col(dbc.Button('Backtest this pair',
                                    id       = 'Backtest_button',
                                    outline  = True,
                                    n_clicks = 0,
                                    style    = dict(margin='10% 0 10% 0',
                                                    color='#75BAF2'))),
                 dbc.Col(width=7)]),

        html.Div(id='load'),

        # Table and graphs of the pair
        dbc.Row(dbc.Col(id='pair_area')),

    ]
)


# ____________________________ CALLBACKS ____________________________
# Update the pairs' dropdown menu based on the selected quote
@app.callback(
    [dash.dependencies.Output('Dropdown_pair', 'options'),
     dash.dependencies.Output('Dropdown_pair', 'value')],
    [dash.dependencies.Input('Dropdown_quote', 'value')])
def update_pair_dropdown(input_quote):
    options = [{'label': pair, 'value': pair} for pair in pairs[input_quote]]
    # value   = pairs[input_quote][0]   # Have the first pair in the list as default
    value   = 'ADABTC'                        # No default
    return options, value

# Update the indicators' dropdown menu based on the selected type of prices
@app.callback(
    dash.dependencies.Output('Dropdown_indicators', 'options'),
    [dash.dependencies.Input('Dropdown_ClosesOrReturns',   'value')])
def update_indicators_dropdown(input_ClosesOrReturns):
    options   = [{'label': 'Trigger points', 'value': 'triggers'}]
    if input_ClosesOrReturns == '':
        options = [{'label': 'Candles', 'value': 'candles'}] + \
                  [{'label': 'Close prices',   'value': 'prices'}] + options
    elif input_ClosesOrReturns == '_log':
        options = [{'label': 'Probabilities of buy points', 'value': 'buys_probas'}] + options
    return options

# # Display "Loading..." when loading
# @app.callback(Output('load',                'children'),
#               [Input('Dropdown_timeframe',  'value'),
#                Input('Dropdown_pair',       'value'),
#                Input('Dropdown_duration',   'value'),
#                # Input('Dropdown_indicators', 'value'),
#                ])
# def prepare_data(input_timeframe, input_pair, nb_triggers):
#     if input_timeframe or input_pair or nb_triggers:
#         return html.Div([dcc.Markdown(
#             '''Loading ...''')], id='pair_area')

# Update the data to display for the pair
@app.callback(
              Output('pair_area',                   'children'),
              [Input('Dropdown_quote',              'value'),
               Input('Dropdown_pair',               'value'),
               Input('Dropdown_ClosesOrReturns',    'value'),
               Input('Dropdown_duration',           'value'),
               Input('Dropdown_indicators',         'value'),
               Input('Backtest_button',             'n_clicks')])
def plot_pair_data(input_quote, input_pair, input_ClosesOrReturns, nb_triggers, input_inds, clicks):
    if input_pair:
        h = plotPairHistory(quoteasset        = input_quote,
                            pair              = input_pair,
                            ClosesOrReturns   = input_ClosesOrReturns,
                            nb_trigger_points = nb_triggers,
                            indicators        = input_inds,
                            clickForBacktest  = clicks)
        return h


if __name__ == "__main__":
    app.run_server(debug=True)