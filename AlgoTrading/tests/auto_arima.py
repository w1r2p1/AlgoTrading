from Exchange     import Binance
from Indicators import *

from datetime  import datetime
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.tsa.stattools as tsa
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pmd
from pmdarima.model_selection import train_test_split
import statsmodels.tsa.statespace.sarimax as sarimax
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.datasets import load_wineind
from pmdarima.datasets import load_msft
from pmdarima.arima.utils import ndiffs
from sklearn.metrics import mean_squared_error
from pmdarima.metrics import smape
from pmdarima.arima.stationarity import ADFTest     # Augmented Dickey-Fuller test to check if the time-series is stationnary
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import os
from os import listdir
from os.path import isfile, join
from pathlib import Path
import pmdarima as pm
import pickle

# quote = 'BTC'
# pair  = 'ELFBTC'

class ArimaFit:


    def __init__(self):

        self.timeframe = '1h'                            # Timeframe in the datasets
        self.train_percentage_allocation = 90            # Percentage of the data set to allocate to training
        self.nb_pred = 100                               # Number of predictions to make
        self.zoom_percentage = 90                        # Percentage of history to plot
        self.exogenous_data_labels = ['open', 'high', 'low', 'EMA_50', 'MACD_12_26', 'MACD_signal_9', 'MACD_histogram', 'BBL_20_2', 'BBM_20_2', 'BBU_20_2']


    def GenerateAllModels(self):
        """ Loop through all the pairs and generate a model for each. """

        # ________________________ DEFINE THE QUOTES & PAIRS TO CREATE MODELS FOR ________________________________________

        # Work on the quotes that have a directory in 'Historical_data/'
        ExistingQuoteassets = next(os.walk('Historical_data/'))[1]                                      # ['ETH', 'BTC']
        timeframe = next(os.walk('Historical_data/' + ExistingQuoteassets[0]))[2][0].split('_')[1]      # '1h'

        # Work on the pairs that have a file in the sub-directory, ie pairs that we have data for.
        pairs = dict()                                                                                  # {'ETH': ['QTUMETH', 'EOSETH',..], 'BTC':[]}
        for quoteasset in ExistingQuoteassets:
            pairs[quoteasset] = [f.replace('_' + timeframe + '_raw', '') for f in listdir('Historical_data/' + quoteasset + '/') if isfile(join('Historical_data/' + quoteasset + '/', f))]

            for pair in pairs[quoteasset]:
                self.GenerateAndSaveModel(quote=quoteasset, pair=pair, timeframe=timeframe)


    def GenerateAndSaveModel(self, quote:str, pair:str, timeframe:str):
        """ Using auto_arima(), find and fit the best model for the pair. Save it to a file for later use. """

        # Get the dataframe from the csv file
        print("_________ MODELING " + pair + " _________")
        filename = 'Historical_data/' + quote + '/' + pair + '_' + timeframe
        df = pd.read_csv(filename, sep='\t')[2000:]                              # remove the first rows of df, where the indicators are at NaN
        del df['buys']
        del df['sells']

        # Divide the dataframe in train & test parts
        train_size = int(self.train_percentage_allocation/100*len(df))
        df_train, df_test = df[:train_size], df[train_size:]

        # Filename to save the model
        arima_filename = 'ARIMA_models/' + quote + '/arima' + pair + '_' + timeframe + '.pkl'

        # Estimate the number of differences to apply using an ADF test (to get stationnay data) :
        n_adf = ndiffs(df['EMA_50'], test='adf')  # -> 1

        # exogenous_data = df[self.exogenous_data_labels]
        # exogenous_data_train = exogenous_data[:train_size]
        # exogenous_data_test  = exogenous_data[train_size:]
        # exogenous_data_train = exogenous_data[:len(exogenous_data)-5]
        # exogenous_data_test  = exogenous_data[len(exogenous_data)-5:]

        # Use auto_arima to find the best p, q & d parameters for the ARIMA model, that minimize AIC
        stepwise_fit = pm.auto_arima(df_train['EMA_50'],
                                     # exogenous          = exogenous_data_train,
                                     test               = 'adf',
                                     d                  = n_adf,         # None : let model determine 'd' : force the ARIMA model to adjust for non-stationarity on its own, without having to worry about doing so manually.
                                     start_p            = 1,
                                     start_q            = 1,
                                     max_p              = 4,
                                     max_q              = 4,
                                     max_iter           = 50,           # The maximum number of function evaluations. Default is 50.
                                     trend              = 'ct',         # a constant and trend term to the equation
                                     # seasonal           = True,
                                     # start_P          = 0,
                                     # D                = 1,            # seasonal differencing term
                                     # m                = 24,           # number of observations per seasonal cycle (must be known apriori)
                                     # max_order          = None,       # Maximum value of p+q+P+Q if model selection is not stepwise.
                                     information_criterion = 'aic',
                                     # alpha              = 0.01,
                                     trace              = True,         # Whether to print status on the fits
                                     error_action       = 'ignore',     # don't want to know if an order does not work
                                     suppress_warnings  = True,         # don't want convergence warnings
                                     stepwise           = True,         # set to stepwise
                                     # n_jobs           = -1,           # How many CPU cores should be used in grid search. -1 = as many as possible
                                     )

        print(stepwise_fit.summary())     # Print the auto_arima whole result
        # stepwise_fit.plot_diagnostics(figsize=(12,10))
        # plt.show()

        # Try to  model an ARIMA based on the order found by auto_arima()
        # order = stepwise_fit.order
        # model = ARIMA(df_train['EMA_50'], order=order)          # (1, 1, 2) for EMA_50
        # fitted_model = model.fit()

        # # In the project directory, create a nested directory for the quoteasset if not exists
        Path('ARIMA_models/' + quote).mkdir(parents=True, exist_ok=True)

        # Save the model
        with open(arima_filename, 'wb') as pkl:            # read 'arima.pkl' : print(pd.read_pickle('arima.pkl'))  ->   ARIMA(order=(1, 2, 1), suppress_warnings=True)
            pickle.dump(stepwise_fit, pkl)


    def PlotArima(self, df_test, forecasts:list):

            fig = make_subplots(rows=2, cols=1,
                                shared_xaxes=True,          # If we zzom on one, we zoom on all
                                row_width=[0.1, 0.9])
            # # Train data
            # fig.add_trace(go.Scatter(x    = df_train.time[int(zoom_percentage/100*len(df_train)):],
            #                          y    = df_train.close[int(zoom_percentage/100*len(df_train)):],
            #                          mode = 'lines',
            #                          name = 'Train data',
            #                          line = dict(color='blue', width=2)),
            #               row=1, col=1)

            # Test data
            fig.add_trace(go.Scatter(x    = df_test.time[-len(forecasts)-50],
                                     y    = df_test.EMA_50[-len(forecasts)-50],
                                     mode = 'lines',
                                     name = 'Test data',
                                     line = dict(color='green', width=2)),
                          row=1, col=1)

            # Plot predicted data
            fig.add_trace(go.Scatter(
                                x    = df_test.time[-len(forecasts):],
                                # x    = df_test.time,
                                y    = forecasts,
                                mode = 'lines',
                                name = 'Predicted data',

                                line = dict(color='red', width=2)),             # chartreuse, chocolate, darkblue, darkgreen, darkorange
                          row=1, col=1)


            # # Plot percentage error
            # fig.add_trace(go.Scatter(
            #     x    = df.time[int(zoom_percentage/100*len(df_train)):],
            #     y    = df['direct_percent_error'][int(zoom_percentage/100*len(df_train)):],
            #     mode = 'lines',
            #     name = 'Percentage error direct',
            #     line = dict(color='darkred', width=2),),
            #     row=2, col=1)


            # percent_max_axes = max(abs(min(direct_percent_error_list)), max(direct_percent_error_list))
            fig.update_layout({
                # "margin": {"t": 30, "b": 20},
                # "height": 800,

                'xaxis1' : {
                    'showline'      : True,
                    'zeroline'      : False,
                    'showgrid'      : False,
                    'showticklabels': True,
                    'rangeslider'   : {'visible': False},
                    'color'         : '#a3a7b0',
                    },
                'yaxis1' : {
                    'fixedrange'    : True,
                    'showline'      : False,
                    'zeroline'      : False,
                    'showgrid'      : False,
                    'showticklabels': True,
                    'ticks'         : '',
                    'color'         : '#a3a7b0',
                    },
                'xaxis2' : {
                    'showline'      : True,
                    'zeroline'      : False,
                    'showgrid'      : False,
                    'showticklabels': True,
                    'rangeslider'   : {'visible': False},
                    'color'         : '#a3a7b0',
                    },
                'yaxis2' : {
                    'title'         : '% Error between prediction and actual',
                    'fixedrange'    : True,
                    'showline'      : False,
                    'zeroline'      : False,
                    'showgrid'      : True,
                    # "dtick"         : 0.5,              # gridline interval
                    'gridwidth'     : 0.1,
                    'gridcolor'     : '#a3a7b0',
                    'showticklabels': True,
                    'ticks'         : '',
                    # 'range'         : [-percent_max_axes, +percent_max_axes]
                    },

                'legend'        : {'font': dict(size=15, color='#a3a7b0'),},
                'plot_bgcolor'  : '#23272c',
                'paper_bgcolor' : '#23272c',
            })

            fig.show()


    def MakePrediction(self, model, npred:int):
        """ Makes predictions over npred future candles. """

        prediction_list = list(model.predict(
                                            n_periods = npred,
                                            # exogenous = exogenous_data,
                                            # return_conf_int = True,
                                        ))
        if npred == 1:
            return prediction_list[0]
        else:
            return prediction_list


    def PredictTestData_AndPlot(self, quote:str, pair:str, nb_forecasts:int):
        """ See how the model goes on test data """

        # Get the dataframe from the csv file
        print("_________ MODELING " + pair + " _________")
        filename = 'Historical_data/' + quote + '/' + pair + '_' + self.timeframe
        df = pd.read_csv(filename, sep='\t')[2000:]                                                                      # remove the first rows of df, where the indicators are at NaN
        del df['buys']
        del df['sells']

        # Divide the dataframe in train & test parts
        train_size = int(self.train_percentage_allocation/100*len(df))
        df_train, df_test = df[:train_size], df[train_size:]

        arima_filename = 'ARIMA_models/' + quote + '/arima' + pair + '_' + self.timeframe + '.pkl'
        # Read the saved model
        with open(arima_filename, 'rb') as pkl:
            model = pickle.load(pkl)

        # Updates the existing model with elapsed data since the model's creation
        model.update(y=df_test.EMA_50[:-nb_forecasts],
                     # exogenous=df_test[self.exogenous_data_labels].iloc[counter],
                     )

        # forecasts = self.MakePrediction(model = model,
        #                                 npred = nb_forecasts,
        #                                 # exogenous_data = df_test[self.exogenous_data_labels].iloc[counter],
        #                                      )
        forecasts = []
        # Iterate over the rest of the test set and predict the next value.
        for counter, new_ob in enumerate(df_test.close[-nb_forecasts:]):
            sys.stdout.write("\r")          # Erase the last line
            sys.stdout.write('Predicting price ' + str(counter) + ' out of ' + str(len(df_test[-nb_forecasts:])))
            # print(df_test[self.exogenous_data_labels].iloc[counter])
            forecasts.append(self.MakePrediction(model = model,
                                                 npred = 1,
                                                 # exogenous_data = df_test[self.exogenous_data_labels].iloc[counter],
                                                 ))
            # Updates the existing model
            model.update(y=new_ob,
                         # exogenous=df_test[self.exogenous_data_labels].iloc[counter],
                         )

        self.PlotArima(df_test=df_test, forecasts=forecasts)





if __name__ == "__main__":
    modelization = ArimaFit()

    modelization.GenerateAndSaveModel(quote='BTC', pair='LTCBTC', timeframe='1h')
    # modelization.PredictTestData_AndPlot(quote = 'BTC', pair = 'LTCBTC', nb_forecasts = 10)