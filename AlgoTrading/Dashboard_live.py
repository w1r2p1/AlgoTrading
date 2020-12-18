from Exchange import Binance
from Database import BotDatabase
from Strategies import *

import datetime
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from decimal import Decimal
import re
import numpy as np
from collections import OrderedDict

exchange = Binance(filename='credentials.txt')
database = BotDatabase("database.db")
ExistingQuoteassets = set([dict(bot)['quoteasset'] for bot in database.GetAllBots()])       # ['ETH', 'BTC']


# ____________________________ QUOTEASSETS ____________________________
def TotalOrders(quoteasset:str)->int:
    """ Returns the number of all the orders ever made on a quoteasset. """
    return len(list(database.GetQuoteassetOrders(quoteasset=quoteasset)))


def RecentOrders(quoteasset:str)->int:
    """ Returns the number of orders we did in the last 24 hours on a quoteasset. """
    return len([dict(order)['pair'] for order in list(database.GetQuoteassetOrders(quoteasset))
                if datetime.utcnow()-timedelta(hours=24) < datetime.strptime(dict(order)['transactTime'], "%Y-%m-%d %H:%M:%S")])


def OpenOrders(quoteasset:str)->int:
    """ Returns the number of currently open buy positions. """
    return len([dict(bot)['pair'] for bot in database.GetAllBots() if dict(bot)['status']=='Looking to exit' if dict(bot)['quoteasset']==quoteasset])


# def Profit(quoteasset:str):
#
#     all_bots = list(database.GetAllBots())
#     profits_list = [Decimal(dict(bot)['bot_profit']) for bot in all_bots if dict(bot)['quoteasset'] == quoteasset if dict(bot)['number_of_orders']>1]
#     cummulative_profit_list = sorted(list(set(np.cumsum(profits_list).tolist())), key=float)
#
#     return cummulative_profit_list[-1]
#
#
# def QuoteFees(quoteasset:str):
#     all_bots = list(database.GetAllBots())
#     quotefees_list = [Decimal(dict(bot)['bot_quote_fees']) for bot in all_bots if dict(bot)['quoteasset'] == quoteasset if dict(bot)['number_of_orders']>0]
#     cummulative_quotefees_list = sorted(list(set(np.cumsum(quotefees_list).tolist())), key=float)
#
#     return cummulative_quotefees_list[-1]
#     # return database.GetQuoteFees(quoteasset=quoteasset)
#
#
# def BNBFees(quoteasset:str):
#     return database.GetBNBFees(quoteasset=quoteasset)
#
#
# def ProfitMinusFees(quoteasset:str):
#     return str(Decimal(Profit(quoteasset)) - Decimal(QuoteFees(quoteasset)))


def Quoteasset_AverageHoldDuration(quoteasset:str):
    """Computes the average holding duration on a quoteasset."""

    hold_durations = []
    for order in list(database.GetQuoteassetOrders(quoteasset=quoteasset)):
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
    # print("Average holding duration on {quoteasset} : {days}d, {hours}h, {minutes}m, {seconds}s.".format(quoteasset = quoteasset,
    #                                                                                                      days       = td.days,
    #                                                                                                      hours      = td.days * 24 + td.seconds // 3600,
    #                                                                                                      minutes    = (td.seconds % 3600) // 60,
    #                                                                                                      seconds    = td.seconds % 60))
    return td


def Pair_RecentOrders(pair:str)->int:
    """ Returns the number of orders we did in the last 24 hours on a pair. """
    return len([dict(order)['pair'] for order in list(database.GetOrdersOfBot(pair))
                if datetime.utcnow()-timedelta(hours=24) < datetime.strptime(dict(order)['transactTime'], "%Y-%m-%d %H:%M:%S")])


def Pair_AverageHoldDuration(pair:str):
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


def plotBalancesEvolution(quoteasset:str, save:bool=False):
    """Plots the evolution of the balance of each quoteasset since the first sell order."""

    fig = dict()

    all_orders = list(database.GetQuoteassetOrders(quoteasset=quoteasset))

    profit_list = [Decimal(dict(order)['profit_minus_fees']) for order in all_orders if dict(order)['side'] == 'SELL']
    cummulative_profit_list1 = np.cumsum(profit_list).tolist()
    # Add the starting balance to each item
    cummulative_profit_list2 = [item + Decimal(database.GetStartBalance(quoteasset)) for item in cummulative_profit_list1]

    fig['data'] = [{'x': [datetime.strptime(dict(order)['transactTime'], '%Y-%m-%d %H:%M:%S') for order in all_orders if
                          dict(order)['side'] == 'SELL'] + [datetime.utcnow()],
                    'y': cummulative_profit_list2 + [cummulative_profit_list2[-1]]}]

    fig['layout'] = {
                     "margin"    : {"t": 30, "b": 20},
                     "title"     : quoteasset + " balance evolution",
                     "titlefont" : dict(size=18,color='#a3a7b0'),
                     "height"    : 350,
                     "xaxis": {
                         # "fixedrange": True,
                         "showline"      : True,
                         "zeroline"      : False,
                         "showgrid"      : False,
                         "showticklabels": True,
                         "color"         : "#a3a7b0"},
                     "yaxis": {
                         # "fixedrange": True,
                         "showline"      : False,
                         "zeroline"      : False,
                         "showgrid"      : True,
                         "showticklabels": True,
                         "ticks"         : "",
                         "color": "#a3a7b0"},
                     "plot_bgcolor" : "#23272c",
                     "paper_bgcolor": "#23272c"}

    # save = True
    # if save:
    #     # # Clear file if exisiting
    #     # with open('plothistory.html', "w"):
    #     #     pass
    #     # Figure = go.Figure(data=fig['data'], layout=fig['layout'])
    #     # Figure.write_html('plothistory.html', include_plotlyjs=False)       # include_plotlyjs : If True, 3MB larger, no internet required to open it
    #
    #     if not os.path.exists("images"):
    #         os.mkdir("images")
    #
    #     Figure = go.Figure(data=fig['data'], layout=fig['layout'])
    #     Figure.write_image("images/fig1.png")

    return fig


# ____________________________ PAIR ____________________________
def plotBotHistory(bot:dict, resample_timeframe:str, nb_last_trades:str, indicators:list):
    """ Plots the candlestick chart with overlays for a bot, since we created it. """

    opening_candles = 20														                            # Number of candles to add at the begining of the plot, just before the bot was created

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    pair = bot['pair']
    # Define the number of candles to plot before the creation of the bot
    parsed_timeframe  = re.findall(r'[A-Za-z]+|\d+', resample_timeframe)      	                            # Separates '30m' in ['30', 'm']

    opening_candles_duration = 0
    if parsed_timeframe[1].lower() == 'm':
        opening_candles_duration = opening_candles * 60     / float(parsed_timeframe[0])
    elif parsed_timeframe[1].lower() == 'h':
        opening_candles_duration = opening_candles * 3600   / float(parsed_timeframe[0])
    elif parsed_timeframe[1].lower() == 'd':
        opening_candles_duration = opening_candles * 86400  / float(parsed_timeframe[0])

    # Set the correct start date of the plot based on the number of trades wanted
    if nb_last_trades == 'all':
        displayed_trades = len(list(database.GetOrdersOfBot(pair)))
        # Get the bot's creation date
        utc_creation_date_string = dict(database.GetBot(pair=pair))['utc_creation_date']				    # datetime string
        utc_creation_date 		 = datetime.strptime(utc_creation_date_string, '%Y-%m-%d %H:%M:%S')			# datetime
        # Compute the starting date of the plot (datetime)
        start_date = utc_creation_date - timedelta(seconds=opening_candles_duration)
        # Plot the bot's creation date as a straight dotted vertical line
        fig.update_layout({
            "shapes"        : [dict(x0=utc_creation_date_string, x1=utc_creation_date_string, y0=0, y1=1, xref='x', yref='paper', line=dict(color="RoyalBlue", width=1, dash="dash"))],
            "annotations"   : [dict(x=utc_creation_date_string, y=0.05, xref='x', yref='paper', showarrow=False, xanchor='left', text='Bot created')],
        })
    else:
        if len(list(database.GetOrdersOfBot(pair))) > int(nb_last_trades):
            #  If enough orders to display
            displayed_trades = int(nb_last_trades)
            first_trade_date_string = dict(list(database.GetOrdersOfBot(pair))[-int(nb_last_trades)])['transactTime']
            start_date              = datetime.strptime(first_trade_date_string, '%Y-%m-%d %H:%M:%S')
        else:
            # If not enough orders, display all of them
            displayed_trades = len(list(database.GetOrdersOfBot(pair)))
            first_trade_date_string = dict(list(database.GetOrdersOfBot(pair))[0])['transactTime']
            start_date              = datetime.strptime(first_trade_date_string, '%Y-%m-%d %H:%M:%S')


    # Get the corresponding candlestick data, from start_date to now
    df = exchange.GetPairKlines(pair=pair, timeframe=resample_timeframe, start_date=start_date)

    # # Obsolete
    # # Re-sample the data to a different timeframe (better for plotting)
    # init_df = exchange.GetPairKlines(pair=pair, timeframe=bot['timeframe'], start_date=start_date)
    # resample_timeframe = '15Min'  # or '1H'
    # init_df.set_index('time')
    # df1 = init_df.resample(resample_timeframe, on='time').agg(OrderedDict([('open',   'first'),
    #                                                                        ('high',   'max'),
    #                                                                        ('low',    'min'),
    #                                                                        ('close',  'last'),
    #                                                                        ('volume', 'sum')]))
    # df  = df1.reset_index()

    if 'candles' in indicators:
        # plot candlestick chart for this pair
        fig.add_trace(go.Candlestick(x 	   = df['time'],
                                     open  = df['open'],
                                     close = df['close'],
                                     high  = df['high'],
                                     low   = df['low'],
                                     name  = "Candlesticks"),
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
                                 line = dict(color='rgb(255,255,51)', width=2)),
                      secondary_y = False)

    # Add the volume
    fig.add_trace(go.Bar(x       = df['time'],
                         y       = df['volume'],
                         name    = "Volume",
                         marker  = dict(color='#a3a7b0')),
                  secondary_y=True)

    # Compute the indicators
    if 'plot_indicators' in indicators:
        strategies_dict[bot['strategy']](df=df, i=len(df['close'])-1, signal='buy', fast_period=50, slow_period=200)

        # Plot the indicators on this pair
        for colname in df.columns[6:]:
            fig.add_trace(go.Scatter(x    = df['time'],
                                     y    = df[colname],
                                     name = colname))

    # Get the all the orders we did on this pair
    orders = database.GetOrdersOfBot(pair=pair)
    buys  = None
    sells = None

    # Buy orders we did on this pair
    if orders is not None:
        buys = [[datetime.strptime(dict(order)['transactTime'], '%Y-%m-%d %H:%M:%S'), Decimal(dict(order)['price'])] for order in list(orders) if dict(order)['side']=='BUY' if datetime.strptime(dict(order)['transactTime'], '%Y-%m-%d %H:%M:%S') >= start_date]
    if buys:
        fig.add_trace(go.Scatter(x 		     = [item[0] for item in buys],
                                 y 			 = [item[1] for item in buys],
                                 name 		 = "Buy orders",
                                 mode 		 = "markers",
                                 marker      = dict(size=10,
                                                    line=dict(width=1.5, color='red'))))

    # Sell orders we did on this pair
    if orders is not None:
        sells = [[datetime.strptime(dict(order)['transactTime'], '%Y-%m-%d %H:%M:%S'), Decimal(dict(order)['price'])] for order in list(orders) if dict(order)['side']=='SELL' if datetime.strptime(dict(order)['transactTime'], '%Y-%m-%d %H:%M:%S') >= start_date]
    if sells:
        fig.add_trace(go.Scatter(x 			 = [item[0] for item in sells],
                                 y 			 = [item[1] for item in sells],
                                 name 		 = "Sell orders",
                                 mode 		 = "markers",
                                 marker      = dict(size=10,
                                                    line=dict(width=1.5, color='green'))))

    fig.update_layout({
        "margin": {"t": 30, "b": 20},
        "height": 800,
        "xaxis" : {
            # "fixedrange"    : True,
            "showline"      : True,
            "zeroline"      : False,
            "showgrid"      : False,
            "showticklabels": True,
            "rangeslider"   : {"visible": False},
            "color"         : "#a3a7b0",
        },
        "yaxis" : {
            "fixedrange"    : True,
            "showline"      : False,
            "zeroline"      : False,
            "showgrid"      : False,
            "showticklabels": True,
            "ticks"         : "",
            "color"         : "#a3a7b0",
        },
        "yaxis2" : {
            "fixedrange"    : True,
            "showline"      : False,
            "zeroline"      : False,
            "showgrid"      : False,
            "showticklabels": True,
            "ticks"         : "",
            # "color"        : "#a3a7b0",
            "range"         : [0, df['volume'].max() * 10],
        },
        "legend" : {
            "font"          : dict(size=15, color="#a3a7b0"),
        },
        "plot_bgcolor"  : "#23272c",
        "paper_bgcolor" : "#23272c",

    })

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>" + pair.replace(bot['quoteasset'], '') + "</b> price in <b>" + bot['quoteasset'] + " </b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Volume</b>", secondary_y=True)


    # Prepare the bootstrap layout for the stats on the pair and the graph

    # Check if the bot sold at least one time and adapt what's displayed
    if len(list(database.GetOrdersOfBot(pair))) > 1:
        profit = dict(database.GetBot(pair=pair))['bot_profit'] + " " + bot['quoteasset']

        profit_minus_fees = "{profit_minus_fees} {quoteasset}".format(quoteasset        = bot['quoteasset'],
                                                                      profit_minus_fees = dict(database.GetBot(pair=pair))['bot_profit_minus_fees'])

        average_hold_duration_string = "{days}d, {hours}h, {minutes}m, {seconds}s.".format(days       = Pair_AverageHoldDuration(pair).days,
                                                                                           hours      = Pair_AverageHoldDuration(pair).days * 24 + Pair_AverageHoldDuration(pair).seconds // 3600,
                                                                                           minutes    = (Pair_AverageHoldDuration(pair).seconds % 3600) // 60,
                                                                                           seconds    = Pair_AverageHoldDuration(pair).seconds % 60)
    else:
        profit                       = "Didn't sell yet"
        profit_minus_fees            = "Didn't sell yet"
        average_hold_duration_string = "Didn't sell yet"


    # Layout below the dropdown menu
    h = html.Div(
                [   # Nom de la pair
                    dbc.Row([
                        dbc.Col(width=1),
                        dbc.Col(dbc.Table(html.Tbody(html.Tr(html.Td(pair, style={'text-align': 'center'}))),
                                          bordered   = False,
                                          borderless = True,
                                          dark       = True,
                                          striped    = True,),
                                width=10),
                        dbc.Col(width=1),
                        ]),

                    # Stats, tableau de gauche
                    dbc.Row([
                        dbc.Col(width=1),
                        dbc.Col(dbc.Table(
                                    html.Tbody(
                                        [
                                            html.Tr([html.Td("Plot Timeframe"),         html.Td(resample_timeframe, style={'color':'#a3a7b0'})]),

                                            html.Tr([html.Td("Bot Timeframe"),          html.Td(bot['timeframe'].replace("m", "Min"), style={'color':'#a3a7b0'})]),

                                            html.Tr([html.Td("Strategy"),               html.Td(bot['strategy'], style={'color':'#a3a7b0'})]),

                                            html.Tr([html.Td("All orders"),             html.Td("{pair_total_orders} (+{recent_orders} in 24h)".format(pair_total_orders=len(list(database.GetOrdersOfBot(pair))), recent_orders=Pair_RecentOrders(pair)), style={'color':'#a3a7b0'})]),

                                            html.Tr([html.Td("Displayed orders"),       html.Td(displayed_trades, style={'color':'#a3a7b0'})]),
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
                                            html.Tr([html.Td("Profit"),                 html.Td(profit, style={'color':'#a3a7b0'})]),

                                            html.Tr([html.Td("Fees"),                   html.Td("{fees_in_quote} {quoteasset} ({fees_in_BNB} BNB)".format(quoteasset    = bot['quoteasset'],
                                                                                                                                                          fees_in_quote = dict(database.GetBot(pair=pair))['bot_quote_fees'],
                                                                                                                                                          fees_in_BNB   = dict(database.GetBot(pair=pair))['bot_BNB_fees']), style={'color':'#a3a7b0'})]),
                                            html.Tr([html.Td("Profit - Fees"),          html.Td(profit_minus_fees, style={'color':'#a3a7b0'})]),

                                            html.Tr([html.Td("Average hold duration"),  html.Td(average_hold_duration_string, style={'color':'#a3a7b0'})]),
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
                ]
    )

    return h


# ____________________________ APP LAYOUT ____________________________
app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                meta_tags=[{"name": "viewport", "content": "width=device-width"}])

app.layout = html.Div(
    [
        html.Div(
            id        = "header",
            children  = [html.Div([html.H2("Bot de moi")], className="eight columns"),
                         html.Div([html.Img(id="logo", src=app.get_asset_url("dash-logo-new.png"))])],
            className = "row"),

        html.Hr(),

        # Loop through the quoteassets and print their stats
        html.Div([html.Div(
                        [
                            dbc.Row(dbc.Col(html.H3(quoteasset + " statistics"))),
                            dbc.Row([
                                    dbc.Col(width=1),
                                    dbc.Col(dbc.Table(
                                                        [
                                                            html.Tbody(
                                                                [
                                                                    html.Tr([html.Td("All orders"),              html.Td("{total_orders} (+{recent_orders} in 24h)".format(total_orders=TotalOrders(quoteasset), recent_orders=RecentOrders(quoteasset)), style={'color':'#a3a7b0'})]),

                                                                    html.Tr([html.Td("Open orders"),             html.Td(OpenOrders(quoteasset), style={'color':'#a3a7b0'})]),

                                                                    html.Tr([html.Td("Binance balance"),         html.Td(exchange.GetAccountBalance(quoteasset) + " " + quoteasset, style={'color':'#a3a7b0'})]),

                                                                    html.Tr([html.Td("Internal balance"),        html.Td(database.GetAccountBalance(quoteasset=quoteasset, real_or_internal='internal') + " " + quoteasset, style={'color':'#a3a7b0'})]),

                                                                    html.Tr([html.Td("Profit"),                  html.Td("{profit_in_quote} {quoteasset} ({profit_in_percentage}%)".format(quoteasset           = quoteasset,
                                                                                                                                                                                           profit_in_quote      = database.GetProfit(quoteasset),
                                                                                                                                                                                           profit_in_percentage = format(round(Decimal(database.GetProfit(quoteasset))/Decimal(database.GetStartBalance(quoteasset=quoteasset))*100, 2), 'f')),
                                                                                                                         style={'color':'#a3a7b0'})]),
                                                                    html.Tr([html.Td("Fees"),                    html.Td("{fees_in_quote} {quoteasset} ({fees_in_BNB} BNB)".format(quoteasset    = quoteasset,
                                                                                                                                                                                   fees_in_quote = database.GetQuoteFees(quoteasset),
                                                                                                                                                                                   fees_in_BNB   = database.GetBNBFees(quoteasset)),
                                                                                                                         style={'color':'#a3a7b0'})]),
                                                                    html.Tr([html.Td("Profit - Fees"),           html.Td("{profit_minus_fees_in_quote} {quoteasset} ({profit_minus_fees_in_quote_in_percentage}%)".format(quoteasset                 = quoteasset,
                                                                                                                                                                                                                          profit_minus_fees_in_quote = database.GetProfit_minus_fees(quoteasset),
                                                                                                                                                                                                                          profit_minus_fees_in_quote_in_percentage = format(round(((Decimal(database.GetStartBalance(quoteasset=quoteasset))+Decimal(database.GetProfit_minus_fees(quoteasset)))/Decimal(database.GetStartBalance(quoteasset=quoteasset))-1)*100, 2), 'f')), style={'color':'#a3a7b0'})]),
                                                                    html.Tr([html.Td("Average hold duration"),   html.Td("{days}d, {hours}h, {minutes}m, {seconds}s.".format(days       = Quoteasset_AverageHoldDuration(quoteasset).days,
                                                                                                                                                                             hours      = Quoteasset_AverageHoldDuration(quoteasset).days * 24 + Quoteasset_AverageHoldDuration(quoteasset).seconds // 3600,
                                                                                                                                                                             minutes    = (Quoteasset_AverageHoldDuration(quoteasset).seconds % 3600) // 60,
                                                                                                                                                                             seconds    = Quoteasset_AverageHoldDuration(quoteasset).seconds % 60), style={'color':'#a3a7b0'})]),
                                                                ]
                                                            )
                                                        ],
                                                        bordered   = False,
                                                        borderless = True,
                                                        dark       = True,
                                                        # hover      = True,
                                                        # responsive = True,
                                                        striped    = True,
                                                    ), width=4),
                                    dbc.Col(dcc.Graph(id="graph-"+str(counter+1), figure=plotBalancesEvolution(quoteasset=quoteasset), config={"doubleClick": "reset"}), width=7),
                                    ]),

                            dbc.Row(dbc.Col(html.Hr()))
                        ])
                 for counter, quoteasset in enumerate(ExistingQuoteassets)]
        ),


        dbc.Row(dbc.Col(html.H3("History of a pair"))),

        # Dropdown menu to select a quote
        dbc.Row([dbc.Col(width=1),
                 dbc.Col(dcc.Dropdown(
                     id          = 'Dropdown_quote',
                     options     = [{'label': quote, 'value': quote} for quote in list(ExistingQuoteassets)],
                     value       = list(ExistingQuoteassets)[0],
                     clearable   = False,
                     placeholder = "Select a quote")),
                 dbc.Col(width=8)]),

        # Dropdown menu to select a pair to plot
        dbc.Row([dbc.Col(width=1),
                 dbc.Col(dcc.Dropdown(
                                id          = 'Dropdown_pair',
                                placeholder = "Select a pair to see its history")),
                 dbc.Col(width=8)]),

        # Dropdown menu to select a timeframe
        dbc.Row([dbc.Col(width=1),
                 dbc.Col(dcc.Dropdown(id          = 'Dropdown_timeframe',
                                      options     = [{'label': '1m',  'value': '1m'},
                                                     {'label': '15m', 'value': '15m'},
                                                     {'label': '30m', 'value': '30m'},
                                                     {'label': '1H',  'value': '1h'},
                                                     {'label': '6H',  'value': '6h'},
                                                     {'label': '1D',  'value': '1d'}],
                                      value       = '6h',
                                      clearable   = False,
                                      )
                         ),
                 dbc.Col(width=8)]),

        # Dropdown menu to select a duration
        dbc.Row([dbc.Col(width=1),
                 dbc.Col(dcc.Dropdown(id        = 'Dropdown_duration',
                                      options   = [{'label': 'Last trade',     'value': '1'},
                                                   {'label': 'Last 5 trades',  'value': '5'},
                                                   {'label': 'Last 10 trades', 'value': '10'},
                                                   {'label': 'Since creation', 'value': 'all'}],
                                      value     = 'all',
                                      clearable = False)),
                 dbc.Col(width=8)]),


        # Dropdown menu to select the indicators to plot
        dbc.Row([dbc.Col(width=1),
                 dbc.Col(dcc.Dropdown(id        = 'Dropdown_indicators',
                                      options   = [{'label': 'Candles',      'value': 'candles'}] +
                                                  [{'label': 'Close prices', 'value': 'prices'}]  +
                                                  [{'label': 'Indicators',   'value': 'plot_indicators'}],
                                      value     = ['prices'],
                                      multi     = True,
                                      clearable = False)),
                 dbc.Col(width=8)]),


        html.Div(id='load'),

        # Graph of the pair
        dbc.Row(dbc.Col(id='pair_area')),

    ])


# ____________________________ CALLBACKS ____________________________
# Update the pairs' dropdown menu based on the selected quote
@app.callback(
    [dash.dependencies.Output('Dropdown_pair',  'options'),
     dash.dependencies.Output('Dropdown_pair',  'value')],
    [dash.dependencies.Input('Dropdown_quote',  'value')])
def update_pair_dropdown(input_quote):
    pairs_list = [dict(bot)['pair'] for bot in database.GetAllBots() if int(dict(bot)['number_of_orders'])>=1 if input_quote in dict(bot)['pair']]
    sorted_pairs_list = sorted(pairs_list)
    options = [{'label': pair, 'value': pair} for pair in sorted_pairs_list]
    value   = ''                        # No default
    return options, value


# # Display "Loading..." when loading
# @app.callback(Output('load', 'children'),
#               [Input('Dropdown_timeframe', 'value'),
#                Input('Dropdown_duration', 'value'),
#                Input('Dropdown_indicators', 'value')
#                ])
# def display_loading(input_timeframe, input_duration, input_inds):
#     if input_timeframe or input_duration or input_inds:
#         return html.Div([dcc.Markdown(
#             '''Loading ...''')], id='pair_graph')


# Update the data to display for the pair
@app.callback(Output('pair_area',           'children'),
              [Input('Dropdown_pair',       'value'),
               Input('Dropdown_timeframe',  'value'),
               Input('Dropdown_duration',   'value'),
               Input('Dropdown_indicators', 'value')])
def plot_pair_data(input_pair, input_timeframe, input_duration, input_inds):
    if input_pair:
        h = plotBotHistory(bot                = dict(database.GetBot(input_pair)),
                           resample_timeframe = input_timeframe,
                           nb_last_trades     = input_duration,
                           indicators         = input_inds)
        return h


if __name__ == "__main__":
    app.run_server(debug=True)
