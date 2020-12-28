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
import pandas as pd

# quote = 'BTC'
# pair  = 'ELFBTC'

class ArimaFit:


    def __init__(self):

        self.timeframe = '1h'                            # Timeframe in the datasets
        self.train_percentage_allocation = 90            # Percentage of the data set to allocate to training
        self.nb_pred = 100                               # Number of predictions to make
        self.zoom_percentage = 99                        # Percentage of history to plot
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
        df = pd.read_csv(filename, sep='\t')[200:]                              # remove the first rows of df, where the indicators are at NaN
        del df['buys']
        del df['sells']

        # Divide the dataframe in train & test parts
        train_size = int(self.train_percentage_allocation/100*len(df))
        df_train, df_test = df[:train_size], df[train_size:]

        # Filename to save the model
        arima_filename = 'ARIMA_models/' + quote + '/arima' + pair + '_' + timeframe + '.pkl'

        # Estimate the number of differences to apply using an ADF test (to get stationnay data) :
        n_adf = ndiffs(df['close'], test='adf')  # -> 0

        exogenous_data = df[self.exogenous_data_labels]
        exogenous_data_train = exogenous_data[:train_size]
        exogenous_data_test  = exogenous_data[train_size:]
        # exogenous_data_train = exogenous_data[:len(exogenous_data)-5]
        # exogenous_data_test  = exogenous_data[len(exogenous_data)-5:]

        # Use auto_arima to find the best p, q & d parameters for the ARIMA model, that minimizes AIC
        model = pm.auto_arima(df_train['close'],
                                     # exogenous          = exogenous_data_train,
                                     d                  = 1,         # None : let model determine 'd' : force the ARIMA model to adjust for non-stationarity on its own, without having to worry about doing so manually.
                                     start_p            = 0,
                                     start_q            = 0,
                                     max_p              = 3,
                                     max_q              = 3,
                                     trend              = 'ct',         # a constant and trend term to the equation
                                     seasonal           = False,
                                     # m                  = 24,
                                     # start_P            = 1,
                                     trace              = True,         # Whether to print status on the fits
                                     error_action       = 'ignore',     # don't want to know if an order does not work
                                     suppress_warnings  = True,         # don't want convergence warnings
                                     stepwise           = True,         # set to stepwise
                                     )

        print(model.summary())     # Print the auto_arima whole result
        model.plot_diagnostics(figsize=(8,8))
        print(model.order)         # Print the model's order only : (0, 1, 0)

        # # In the project directory, create a nested directory for the quoteasset if not exists
        Path('ARIMA_models/' + quote).mkdir(parents=True, exist_ok=True)

        # Save the model
        with open(arima_filename, 'wb') as pkl:            # read 'arima.pkl' : print(pd.read_pickle('arima.pkl'))  ->   ARIMA(order=(1, 2, 1), suppress_warnings=True)
            pickle.dump(model, pkl)


    def PlotArima(self, df_train, df_test, forecasts:list):

        fig = go.Figure()

        # Train data
        fig.add_trace(go.Scatter(x    = df_train.time[int(self.zoom_percentage/100*len(df_train)):],
                                 y    = df_train.close[int(self.zoom_percentage/100*len(df_train)):],
                                 mode = 'lines',
                                 name = 'End of Train data',
                                 line = dict(color='blue', width=2)))

        # Test data
        fig.add_trace(go.Scatter(x    = df_test.time[:len(forecasts)],
                                 y    = df_test.close[:len(forecasts)],
                                 mode = 'lines',
                                 name = 'Test data',
                                 line = dict(color='green', width=2)))

        # Plot predicted data
        fig.add_trace(go.Scatter(
                            x    = df_test.time[:len(forecasts)],
                            y    = forecasts,
                            mode = 'lines',
                            name = 'Predicted data',
                            line = dict(color='red', width=2)))

        fig.update_layout({
            'xaxis' : {
                'showline'      : True,
                'zeroline'      : False,
                'showgrid'      : False,
                'showticklabels': True,
                'rangeslider'   : {'visible': False},
                'color'         : '#a3a7b0',
                },
            'yaxis' : {
                'fixedrange'    : True,
                'showline'      : False,
                'zeroline'      : False,
                'showgrid'      : False,
                'showticklabels': True,
                'ticks'         : '',
                'color'         : '#a3a7b0',
                },

            'legend'        : {'font': dict(size=15, color='#a3a7b0'),},
            'plot_bgcolor'  : '#23272c',
            'paper_bgcolor' : '#23272c',
        })

        fig.show()


    def MakePrediction(self, model, npred:int):
        """ Makes predictions over npred future candles. """

        predictions = list(model.predict(n_periods = npred,
                                         # typ = 'levels',
                                        # exogenous = exogenous_data,
                                        # return_conf_int = True,
                                        ))
        if npred == 1:
            return predictions[0]
        else:
            return predictions


    def PredictTestData_AndPlot(self, quote:str, pair:str, df_train, df_test, nb_forecasts:int):
        """ See how the model goes on test data """

        # Get the dataframe from the csv file
        print("_________ MODELING " + pair + " _________")
        filename = 'Historical_data/' + quote + '/' + pair + '_' + self.timeframe
        df = pd.read_csv(filename, sep='\t')[200:]                                                                      # remove the first rows of df, where the indicators are at NaN
        del df['buys']
        del df['sells']

        # Filename to save the model
        arima_filename = 'ARIMA_models/' + quote + '/arima' + pair + '_' + self.timeframe + '.pkl'
        # Read the saved model
        with open(arima_filename, 'rb') as pkl:
            model = pickle.load(pkl)

        # forecasts = self.MakePrediction(model = model,
        #                                 npred = nb_forecasts,
        #                                 )


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

        self.PlotArima(df_train=df_train, df_test=df_test, forecasts=forecasts)





if __name__ == "__main__":
    modelization = ArimaFit()

    # modelization.GenerateAndSaveModel(quote='BTC', pair='ADABTC', timeframe='1h')
    df = pd.read_csv('Historical_data/BTC/ADABTC_1h', sep='\t')[200:]                                 # remove the first rows of df, where the indicators are at NaN
    modelization.PredictTestData_AndPlot(quote        = 'BTC',
                                         pair         = 'ADABTC',
                                         df_train     = df[:int(modelization.train_percentage_allocation/100*len(df))],
                                         df_test      = df[int(modelization.train_percentage_allocation/100*len(df)):],
                                         nb_forecasts = 20)