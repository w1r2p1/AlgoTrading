from Exchange     import Binance
from Indicators import *

import warnings
from datetime  import datetime
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.tsa.stattools as tsa
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pmd
from pmdarima.model_selection import train_test_split
from statsmodels.tsa.arima_model import ARIMA
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
import itertools
import pandas as pd

plt.style.use('fivethirtyeight')

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
        df = pd.read_csv(filename, sep='\t')[200:]                              # remove the first rows of df, where the indicators are at NaN
        del df['buys']
        del df['sells']

        # Divide the dataframe in train & test parts
        train_size = int(self.train_percentage_allocation/100*len(df))
        df_train, df_test = df[:train_size], df[train_size:]

        # Filename to save the model
        arima_filename = 'ARIMA_models/' + quote + '/arima' + pair + '_' + timeframe + '.pkl'

        # Estimate the number of differences to apply using an ADF test (to get stationnay data) :
        n_adf = ndiffs(df['close'], test='adf')  # -> 1




        # Define the p, d and q parameters to take any value between 0 and 2
        p = d = q = range(0, 2)
        # Generate all different combinations of p, q and q triplets
        pdq = list(itertools.product(p, d, q))
        warnings.filterwarnings("ignore") # specify to ignore warning messages

        # Define the p, d and q parameters to take any value between 0 and 2
        p = d = q = range(0, 2)

        # Generate all different combinations of p, q and q triplets
        pdq = list(itertools.product(p, d, q))

        warnings.filterwarnings("ignore") # specify to ignore warning messages

        c4=[]
        for param in pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(df_train.close,
                                                order=param,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)

                results = mod.fit()

                print('ARIMA{} - AIC:{}'.format(param, results.aic))
                c4.append('ARIMA{} - AIC:{}'.format(param, results.aic))
            except:
                continue

        print(c4)




        # # # In the project directory, create a nested directory for the quoteasset if not exists
        # Path('ARIMA_models/' + quote).mkdir(parents=True, exist_ok=True)

        # # Save the model
        # with open(arima_filename, 'wb') as pkl:            # read 'arima.pkl' : print(pd.read_pickle('arima.pkl'))  ->   ARIMA(order=(1, 2, 1), suppress_warnings=True)
        #     pickle.dump(mod, pkl)


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
            fig.add_trace(go.Scatter(x    = df_test.time,
                                     y    = df_test.close,
                                     mode = 'lines',
                                     name = 'Test data',
                                     line = dict(color='green', width=2)),
                          row=1, col=1)

            # Plot predicted data
            fig.add_trace(go.Scatter(
                                x    = df_test.time[-len(forecasts):],
                                y    = forecasts,
                                mode = 'lines',
                                name = 'Predicted data direct',

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

        prediction_list = list(model.predict(n_periods = npred,
                                        # exogenous = exogenous_data,
                                        # return_conf_int = True,
                                        ))

        if npred == 1:
            return prediction_list[0]
        else:
            return prediction_list


    def PredictTestData_AndPlot(self, quote:str, pair:str, df_test, nb_forecasts:int):
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

        # Updates the existing model with elapsed data since the model's creation
        model.update(y=df_test.close[:-nb_forecasts],
                     # exogenous=df_test[self.exogenous_data_labels].iloc[counter],
                     )

        forecasts = self.MakePrediction(model = model,
                                        npred = nb_forecasts,
                                        # exogenous_data = df_test[self.exogenous_data_labels].iloc[counter],
                                        )




        self.PlotArima(df_test=df_test, forecasts=forecasts)





if __name__ == "__main__":
    modelization = ArimaFit()

    modelization.GenerateAndSaveModel(quote='BTC', pair='ADABTC', timeframe='1h')
    # df = pd.read_csv('Historical_data/BTC/ADABTC_1h', sep='\t')[200:]                                 # remove the first rows of df, where the indicators are at NaN
    # modelization.PredictTestData_AndPlot(quote        = 'BTC',
    #                                      pair         = 'ADABTC',
    #                                      df_test      = df[int(modelization.train_percentage_allocation/100*len(df)):],
    #                                      nb_forecasts = 100)