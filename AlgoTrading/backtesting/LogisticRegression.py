import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle
import math
import matplotlib.pyplot as plt
from pathlib  import Path
import sys
import string
import re
import seaborn as sns
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from tqdm import tqdm
from rtchange import Finder



class ModelSelection:

    def __init__(self, quote, pair, timeframe, side, log):

        self.quote = quote
        self.pair  = pair
        self.side  = side
        self.log   = log
        self.timeframe = timeframe

        self.df = self.get_df()
        self.histos_dict = self.histograms_from_CSV()

        self.momentum = dict(
            ao          = (5,20),
            apo         = (5,20),
            cci         = (30,),
            cmo         = (30,),
            coppock     = (10,11,14),
            er          = (30,),
            fisher      = (9,1),
            inertia     = (30,14),
            kdj         = (30,),
            macd        = (10,30,9),
            mom         = (30,),
            pgo         = (30,),
            psl         = (30,),
            pvo         = (10,30,9),
            slope       = (1,),
            tsi         = (15,30),
            willr       = (30,),
            eri         = (30,),
            bop         = tuple(),
            squeeze     = tuple(),
            uo          = tuple(),
            # rsi       = (30,),          # A voir, remove first outlier
            # smi       = (10,30,9),      # needs package update to work
            # stochrsi  = (30,30),        # needs package update to work
            # bias      = (30,),          # 10e6  values and gives 1 bar
            # brar      = (30,),          # 10e15 values and gives 1 bar
            # rvgi      = (30,15),        # 10e5  values and gives 1 bar
            # stoch     = (15,5,5),       # Gives 1 bar
            # trix      = (30,15),        # Gives 1 bar
            # cg        = (30,),          # Doesn't work
            # ppo       = (10,30,9),      # Doesn't work
            # roc       = (30,),          # Doesn't work

        )
        self.overlap = dict(
            dema        = (30,),
            ema         = (30,),
            fwma        = (30,),
            hma         = (30,),
            linreg      = (30,),
            midpoint    = (2,),
            pwma        = (30,),
            rma         = (30,),
            t3          = (30,),
            tema        = (30,),
            trima       = (30,),
            vwma        = (30,),
            wma         = (30,),
            zlma        = (30,),
            hl2         = tuple(),
            hlc3        = tuple(),
            ohlc4       = tuple(),
            ssf         = (30,3),
            vwap        = tuple(),
            wcp         = tuple(),
            # hilo      = (15,30),        # needs package update to work
            # kama      = (15,10,30),     # Doesn't work
        )
        """
        Kurtosis (KURT)
        https://www.spcforexcel.com/knowledge/basic-statistics/are-skewness-and-kurtosis-useful-statistics
        Measure of the combined weight of the tails relative to the rest of the distribution, on both sides. Assumes a normal distribution !!
        Distributions with large kurtosis exhibit tail data exceeding the tails of the normal distribution.
        High kurtosis of the return distribution implies that the investor will experience occasional extreme returns (either positive or negative).
        .kurt() : Kurtosis obtained using Fisher’s definition of kurtosis (kurtosis of normal == 0.0). Normalized by N-1.
        kurtosis = close.rolling(length, min_periods=min_periods).kurt()
        
        Mean Absolute Deviation (MAD)
        Average distance between each data value and the mean
        def mad_(series):
        from numpy import fabs as npfabs
        return npfabs(series - series.mean()).mean()
        mad = close.rolling(length, min_periods=min_periods).apply(mad_, raw=True)
        
        Median (MEDIAN)
        Median value of the last X values
        median = close.rolling(length, min_periods=min_periods).median()
        
        Quantile (QTL_{length}_{q})
        quantile = close.rolling(length, min_periods=min_periods).quantile(q)
        
        Skew (SKEW)
        https://www.spcforexcel.com/knowledge/basic-statistics/are-skewness-and-kurtosis-useful-statistics
        Skewness is usually described as a measure of a dataset’s symmetry – or lack of symmetry.
        A perfectly symmetrical data set will have a skewness of 0. The normal distribution has a skewness of 0.
        skew = close.rolling(length, min_periods=min_periods).skew()
        
        stdev  (STDEV_{length})
        stdev = variance(close=close, length=length).apply(npsqrt)
        
        Z Score  (Z_{length})
        Computation :
        std    = stdev(close=close, length=length, **kwargs)
        mean   = sma(close=close, length=length, **kwargs)
        zscore = (close - mean) / std
        
        Entropy (ENTP)
        Computation :
        Higher entropy = higher unpredictability
        p = close / close.rolling(length).sum()
        entropy = (-p * npLog(p)).rolling(length).sum()
        
        Variance  (VAR_{length})
        variance = close.rolling(length, min_periods=min_periods).var()
        
        """
        self.statistics = dict(
            kurtosis    = (30,),
            mad         = (30,),
            median      = (30,),
            quantile    = (30,),
            skew        = (30,),
            stdev       = (2,),
            zscore      = (30,),
            variance    = (30,),
            # entropy   = (30,),          # Doesn't work
        )
        self.trend = dict(
            adx            = (20,),
            amat           = (5,20),
            aroon          = (20,),
            chop           = (20,),
            cksp           = (10,1,9),
            decreasing     = (20,),
            dpo            = (20,),
            increasing     = (20,),
            qstick         = (20,),
            supertrend     = (7,3),
            vortex         = (20,),
            # linear_decay = (20,),       # needs package update to work
            # long_run     = (20,),       # No fast and slow columns in df
            # short_run    = (20,),       # No fast and slow columns in df
            # psar         = (0.02,0.2),  # Doesn't work
        )
        """
        Acceleration Bands (ACCBANDS)
            Returns 3 dataframe columns : ACCBL_{length}, ACCBM_{length}, ACCBU_{length}
            https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/acceleration-bands-abands/
            Acceleration Bands created by Price Headley plots upper and lower envelope bands around a simple moving average.
            HL_RATIO = c * (high - low) / (high + low)
            LOW      = low * (1 - HL_RATIO)
            HIGH     = high * (1 + HL_RATIO)
            LOWER    = EMA(LOW, length)
            MID      = EMA(close, length)
            UPPER    = EMA(HIGH, length)
        
        Average True Range (ATR)
            Average True Range is used to measure volatility, especially volatility caused by gaps or limit moves.
            ATR = EMA(TR(high, low, close), length)
            Taken as the GREATEST of the following:
            1) current high - current low
            2) absolute value of current high - previous close
            3) absolute value of current low  - previous close
        
        Bollinger Bands (BBANDS)
            Returns 3 dataframe columns : BBL_{length}, BBM_{length}, BBU_{length}
            https://www.tradingview.com/wiki/Bollinger_Bands_(BB)
            LOWER = MID - nb_std*stdev
            MID   = EMA(close, length)
            UPPER = MID + nb_std*stdev
        
        Donchian Channels (DC)
            Returns 3 dataframe columns : DCL_{lower_length}_{upper_length}, DCM_{lower_length}_{upper_length}, DCU_{lower_length}_{upper_length}
            https://www.investopedia.com/terms/d/donchianchannels.asp
            LOWER = close.rolling(length).min()
            MID   = 0.5 * (LOWER + UPPER)
            UPPER = close.rolling(length).max()
        
        Keltner Channels (KC)
            Returns 3 dataframe columns : KCL_{length}, KCB_{length}, KCU_{length}
            https://www.tradingview.com/support/solutions/43000502266-keltner-channels-kc/
            Lower Envelope = 20 Period EMA - (scalar X ATR)
            Basis          = 20 Period EMA
            Upper Envelope = 20 Period EMA + (scalar X ATR)
        
        Mass Index (MASSI)
            Non-directional volatility indicator that utilitizes the High-Low Range to identify trend reversals based on range expansions.
            It suggests that a reversal of the current trend will likely take place when the range widens beyond a certain point and then contracts.
        
            hl_ema1 = ema(close=high_low_range, length=fast, **kwargs)
            hl_ema2 = ema(close=hl_ema1, length=fast, **kwargs)
            hl_ratio = hl_ema1 / hl_ema2
            massi = hl_ratio.rolling(slow, min_periods=slow).sum()
        
        Normalized Average True Range (NATR)
            Normalized Average True Range.
            NATR = (100 / close) * ATR(high, low, close)
        
        Relative Volatility Index (RVI)
            Returns 3 dataframe columns : KCL_{length}, KCB_{length}, KCU_{length}
            https://www.tradingview.com/support/solutions/43000502266-keltner-channels-kc/
            Lower Envelope = 20 Period EMA - (scalar X ATR)
            Basis          = 20 Period EMA
            Upper Envelope = 20 Period EMA + (scalar X ATR)
        """
        self.volatility = dict(
            aberration  = (5,15),
            accbands    = (20,4),
            bbands      = (20,4),
            kc          = (20,),
            # atr       = (20,),
            # natr      = (20,),
            # massi     = (9,25),
        )
        self.volume = dict(
            adosc       = (5,10),
            aobv        = (5,10),
            cmf         = (None, 20),
            efi         = (20,),
            eom         = (20,),
            nvi         = (1,),
            pvi         = (1,),
            vp          = (10,),
            ad          = tuple(),
            obv         = tuple(),
            pvol        = tuple(),
            pvt         = tuple(),
        )

        self.all_custom_inds = {**self.momentum, **self.overlap, **self.statistics, **self.trend, **self.volatility}


    def compute_indicators(self, df):
        """ Computes and adds the indicators to self.df """

        # # Work with the indicators computed by Creation_Histogrammes.py
        # ind_list = self.histos_dict.keys()
        # print("\tComputing the {inds} indicators".format(inds=len(ind_list)))
        # ta_strat = [{'kind':ind.split('_')[0].lower().rstrip(string.digits),
        #              # 'params':tuple([float(ind.split('_')[i]) if '.' in ind.split('_')[i] or 'e' in ind.split('_')[i].lower() else int(ind.split('_')[i]) for i in range(1, len(ind.split('_')))]), # Convert the params to float or int depending if '.' is in the param.
        #              'params':tuple([None if i=='None' else (float(i) if '.' in i or 'e' in i.lower() else int(i)) for i in ind.split('_')[1:]]),       # Convert the params to float or int depending if '.' is in the param.
        #              'col_numbers':(int(float(ind.split('_')[0].replace(ind.split('_')[0].lower().rstrip(string.digits), ''))),) if ind.split('_')[0].replace(ind.split('_')[0].lower().rstrip(string.digits), '') != '' else ''}   # One column at a time here. rstrip(string.digits) :  drop trailling digits
        #             for ind in ind_list]

        # Work with all the indicators defined in the __init__
        print(f"\tComputing the {len(self.all_custom_inds.keys())} indicators")
        ta_strat = [{'kind'     : ind,
                     'params'   : params,
                     'col_names': tuple(["{name}_{params}".format(name=f'{ind}{str(i)}', params='_'.join(list(map(str,params))) if len(params)>1 else params[0] if len(params)==1 else 'None') for i in range(5)])
                     # 'col_names': ind,
                     }
                    for ind, params in self.all_custom_inds.items()]

        # Create a custom strategy object containing the indicators to compute
        CustomStrategy = ta.Strategy(name = "MyStrat",
                                     ta   = ta_strat,
                                     )
        df.ta.cores = 6

        # Run the CustomStrategy on the df : compute the indicators
        df.ta.strategy(CustomStrategy,
                      # timed=True,
                      # verbose=True
                      )

        # Remove the first 100 rows to avoid dealing with NaNs
        # df.drop(df.index[:100], inplace=True)
        # df.reset_index(drop=True, inplace=True)

        # with pd.option_context('display.max_rows', 10, 'display.max_columns', None):
        #     print(df)

        return df


    def convert_to_probabilities(self, df):
        """ Convert the indicators' values to their probability of being a trigger. """

        ind_list = self.histos_dict.keys()

        print("\tConverting the indicators to their associated probabilities. ", end='')

        # Convert the indicators values to their corresponding probabilities, based on their histograms
        df_probas_temp = df.copy()
        for i, ind in enumerate(ind_list):
            for point in tqdm(range(len(df_probas_temp.close))):
                # Last realization of the indicator
                ind_rea = df_probas_temp.iloc[point, df_probas_temp.columns.get_loc(ind)]
                # Find the bin that the value falls in and get the corresponding probability
                try:
                    df_probas_temp.iloc[point, df_probas_temp.columns.get_loc(ind)] = self.histos_dict[ind].loc[ind_rea]
                except KeyError:
                    # df_probas_temp.iloc[point, df_probas_temp.columns.get_loc(ind)] = float('nan')		                # When a realization is not in any bin of the histogram
                    df_probas_temp.iloc[point, df_probas_temp.columns.get_loc(ind)] = 0				                        # When a realization is not in any bin of the histogram


        # Drop the rows where at least one probability is at NaN
        df_probas = df_probas_temp.dropna(subset=ind_list).reset_index(drop=True)

        # Check the rows that we just deleted
        rows_with_na = df_probas_temp[~df_probas_temp.index.isin(df_probas_temp.dropna(subset=ind_list).index)]
        print("\tDeleted {deleted_rows} rows that contained NaN(s), out of {total_rows} ({ratio}%). Remaining rows : {remaining}.".format(deleted_rows   = len(rows_with_na),
                                                                                                                                          total_rows     = len(df_probas_temp),
                                                                                                                                          ratio          = round(len(rows_with_na)/len(df_probas_temp)*100, 1),
                                                                                                                                          remaining      = len(df_probas),
                                                                                                                                          ))
        return df_probas


    def fit_save__and_evaluate_model(self, sklearn_model:str):
        """Fits the model and saves it. """

        # ind_list  = self.histos_dict.keys()

        # # Work with the indicators computed by Creation_Histogrammes.py
        # df_probas = self.convert_to_probabilities(self.compute_indicators(self.df))

        # Work with all the indicators defined in the __init__
        df_probas = self.compute_indicators(self.df).dropna(axis='columns').dropna(axis='index')

        print(f"\tFitting a {sklearn_model} model")

        # Fit and work on a smaller portion of the df
        max_ = int(len(df_probas)*0.75)
        df_pour_fit = df_probas.iloc[:max_].dropna(axis='columns')
        X = df_pour_fit[list(df_pour_fit.columns[7:])]
        Y = df_pour_fit[self.side]

        # Split the data with stratification
        # X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.3, random_state=7, stratify=Y)       # random_state : Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls

        # Fit the model on the training set
        def fit_model():

            if sklearn_model == 'LogisticRegression' :
                # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
                model = LogisticRegression(C        = 1000000,
                                           max_iter = 1000,             # default = 100
                                           solver   = 'liblinear',      # For small datasets, ‘liblinear’ is a good choice
                                           fit_intercept = False,
                                           )
                model.fit(X, Y)
                # print("\tLogistic Regression model fitted and saved ! Coefficients : \n", pd.concat([pd.DataFrame(X_train.columns),pd.DataFrame(np.transpose(model.coef_[0]))], axis = 1))

                return model

            if sklearn_model == 'RandomForest' :
                # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
                model_RF = RandomForestClassifier(n_estimators = 1000)         # default = 100
                # param_grid = {
                #     'bootstrap': [False, True],
                #     'max_depth': [80, 90, 100, 110],
                #     'max_features': [2, 3],
                #     'min_samples_leaf': [3, 4, 5],
                #     'min_samples_split': [8, 10, 12],
                #     'n_estimators': [100, 200, 300, 1000]
                #     }
                model_RF.fit(X, Y)

                # Plot the features importances
                features    = X.columns
                importances = model_RF.feature_importances_
                indices     = np.argsort(importances)
                plt.figure(figsize = (16,12))
                plt.title('Random Forest - Feature Importances')
                plt.barh(range(len(indices)), importances[indices], color='b', align='center')
                plt.yticks(range(len(indices)), [features[i] for i in indices])
                plt.xlabel('Relative Importance')
                plt.show()

                return model_RF

            if sklearn_model == 'DecisionTree' :
                # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
                model_ = DecisionTreeClassifier().fit(X, Y)
                return model_

            if sklearn_model == 'MLP' :
                # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
                model = MLPClassifier(solver='lbfgs',
                                      alpha=1e-5,
                                      # hidden_layer_sizes=(5, 2),
                                      random_state=1,
                                      max_iter=10000,    # default=200
                                      )
                model.fit(X, Y)

                return model

            if sklearn_model == 'KNeighbors' :
                # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

                # # List Hyperparameters that we want to tune.
                # leaf_size   = list(range(1,50))
                # n_neighbors = list(range(1,30))
                # p           = [1,2]
                # # algorithm   = ['auto', 'ball_tree', 'kd_tree', 'brute']
                # # Convert to dictionary
                # hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
                # # Use GridSearch
                # model = GridSearchCV(KNeighborsClassifier(weights='distance'), hyperparameters, cv=10)

                # # Print The value of best Hyperparameters
                # print('Best leaf_size:',    model.get_params()['leaf_size'])
                # print('Best p:',            model.get_params()['p'])
                # print('Best n_neighbors:',  model.get_params()['n_neighbors'])
                # print('Best algorithm:',    best_model.get_params()['algorithm'])

                model_knn = KNeighborsClassifier(weights='distance',              # default = 'uniform'
                                                 n_neighbors = 5)

                return model_knn.fit(X, Y)

        # Save the model to disk
        # model_file = f'models/{self.quote}/{self.timeframe}/{self.pair}_{self.timeframe}_model_{self.side}.sav'
        # pickle.dump(fit_model(), open(model_file, 'wb'))


        # some time later... _______________________________________________________________________________________________________________________________________________________

        # load the model from disk
        # loaded_model = pickle.load(open(model_file, 'rb'))
        loaded_model = fit_model()

        # Estimate the accuracy on the train set
        df_pour_fit.loc[:,'predictions']      = loaded_model.predict(X)
        # df_pour_fit.loc[:,'zero_probability'] = loaded_model.predict_proba(X)[:,0]
        # df_pour_fit.loc[:,'one_probability']  = loaded_model.predict_proba(X)[:,1]
        df_pour_fit.loc[:,'predictions_adjusted'] = np.where(df_pour_fit['predictions']==1, df_pour_fit['close'], np.nan)
        df_pour_fit.loc[:,'truth_adjusted']       = np.where(df_pour_fit[self.side]==1,     df_pour_fit['close'], np.nan)

        # Estimate the accuracy on the test set
        df_outofsample = df_probas.iloc[max_:]
        X_outpofsample = df_outofsample[list(df_outofsample.columns[7:])]
        Y_outpofsample = df_outofsample[self.side].astype(int)
        df_outofsample.loc[:,'outofsample_predictions'] = loaded_model.predict(X_outpofsample)
        df_outofsample.loc[:,'outofsample_predictions_adjusted'] = np.where(df_outofsample['outofsample_predictions']==1, df_outofsample['close'], np.nan)
        df_outofsample.loc[:,'outofsample_truth_adjusted']       = np.where(df_outofsample[self.side]==1,                 df_outofsample['close'], np.nan)

        # Print some metrics
        counts_truth = df_outofsample.loc[:,self.side].value_counts().to_dict()
        nb_truth = counts_truth[list(counts_truth.keys())[1]]
        nb_detec = getattr(df_outofsample.loc[:,'outofsample_predictions'].value_counts().to_dict(), str(1.0), 0)
        print("\tPredicted {a} {side} out of {b} ({det}%). In numbers, not precision".format(det=round(nb_detec/nb_truth*100,1), a=nb_detec, b=nb_truth, side=self.side))
        print("\tAccuracy = ", accuracy_score(df_outofsample.loc[:,self.side], df_outofsample.loc[:,'outofsample_predictions']))

        # ________________________________________________________________________________________________________

        plot = True
        if plot :
            fig  = go.Figure()
            min_ = 0
            max_ = len(df_probas.index)

            # plot the prices or log returns for this pair
            fig.add_trace(go.Scatter(x    = df_pour_fit.index[:max_],
                                     y    = df_pour_fit['close'][:max_],
                                     mode = 'lines',
                                     name = 'Log returns',
                                     ))
            fig.update_yaxes(title_text  = "<b>" + self.pair.replace(self.quote, '') + "</b> log returns")
            fig.update_layout({"yaxis" : {"zeroline" : True},
                               "title" : f'{sklearn_model} - {self.side} predictions and truth.\nCustom {round(nb_detec/nb_truth*100,1)}% accuracy.'})


            # Add the train pivots
            fig.add_trace(go.Scatter(x    = df_pour_fit.index[:max_],
                                     y    = df_pour_fit['truth_adjusted'][:max_],
                                     mode   = "markers",
                                     marker = dict(size   = 9,
                                                   color  = 'green',
                                                   symbol = 'cross',
                                                   ),
                                     name = f'{self.side} pivots'))

            # Add the predicted train pivots
            fig.add_trace(go.Scatter(x    = df_pour_fit.index[:max_],
                                     y    = df_pour_fit['predictions_adjusted'][:max_],
                                     mode   = "markers",
                                     marker = dict(size   = 10,
                                                   color  = 'red',
                                                   symbol = 'circle-open',
                                                   ),
                                     name = f'Predicted {self.side} pivots'))

            # Out of sample _______________
            # prices or Log returns
            fig.add_trace(go.Scatter(x    = df_outofsample.index,
                                     y    = df_outofsample['close'],
                                     mode = 'lines',
                                     name = 'Out of sample Log returns',
                                     ))
            # Add the test pivots
            fig.add_trace(go.Scatter(x    = df_outofsample.index,
                                     y    = df_outofsample['outofsample_truth_adjusted'],
                                     mode   = "markers",
                                     marker = dict(size   = 10,
                                                   color  = 'blue',
                                                   symbol = 'cross',
                                                   ),
                                     name = f'Out of sample {self.side} pivots'))

            # Add the predicted test pivots
            fig.add_trace(go.Scatter(x    = df_outofsample.index,
                                     y    = df_outofsample['outofsample_predictions_adjusted'],
                                     mode   = "markers",
                                     marker = dict(size   = 10,
                                                   color  = 'red',
                                                   symbol = 'circle-open',
                                                   ),
                                     name = f'Predicted {self.side} pivots'))

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


    def test_RTChange(self):
        # f = Finder(discounting_param=0.2, order=3, smoothing=3)
        # self.df.loc[:,'scores'] = list(f.score(self.df.close))

        plt.figure(figsize=(12,15))

        # Scores on the prices
        f1 = Finder(discounting_param=0.2, order=4, smoothing=10)
        self.df.loc[:,'scores_prices'] = list(f1.score(self.df.close))
        plt.subplot(511)
        plt.title("Prices")
        plt.plot(self.df.index, self.df.close, color='#4DB6AC')

        self.df.loc[:,'seuille_ajuste_prices'] = np.where(self.df.scores_prices>9, self.df.close, np.nan)
        plt.scatter(self.df.index, self.df.seuille_ajuste_prices, color='red', marker="x")
        plt.subplot(512)
        plt.title("Change scores of prices")
        plt.plot(self.df.index, self.df.scores_prices, color='#FF5252')

        # Scores on the log returns, also translated in prices
        f2 = Finder(discounting_param=0.2, order=4, smoothing=10)
        self.df.loc[:,'scores_log_returns'] = list(f2.score(self.df.close_log))
        plt.subplot(513)
        plt.title("Log returns")
        plt.plot(self.df.index, self.df.close_log, color='#4DB6AC')
        self.df.loc[:,'seuille_ajuste_log_returns'] = np.where(self.df.scores_log_returns>9, self.df.close_log, np.nan)
        plt.scatter(self.df.index, self.df.seuille_ajuste_log_returns, color='red', marker="x")
        plt.subplot(514)
        plt.title("Prices with points from the log returns")
        plt.plot(self.df.index, self.df.close, color='#4DB6AC')
        self.df.loc[:,'seuille_ajuste_prices2'] = np.where(self.df.scores_log_returns>9, self.df.close, np.nan)
        plt.scatter(self.df.index, self.df.seuille_ajuste_prices2, color='red', marker="x")
        plt.subplot(515)
        plt.title("Change scores of log returns")
        plt.plot(self.df.index, self.df.scores_log_returns, color='#FF5252')

        plt.show()


    def get_df(self):
        """ Gets the historic data from the csv file """

        # # Work on historical data from the csv file
        historical_data_file = f'../historical_data/{self.quote}/{self.timeframe}/{self.pair}_{self.timeframe}'
        dataframe_ = pd.read_csv(historical_data_file, sep='\t')

        # Remove duplicated lines in the historical data if present
        dataframe = dataframe_.loc[~dataframe_.index.duplicated(keep='first')]

        # print(((len(dataframe.buys.dropna())+len(dataframe.sells.dropna()))/len(dataframe.close.dropna()))*100, '%')

        if self.log:
            # Compute the log returns
            dataframe.loc[:,'open']  = np.log(dataframe.loc[:,'open'].pct_change()+1)
            dataframe.loc[:,'high']  = np.log(dataframe.loc[:,'high'].pct_change()+1)
            dataframe.loc[:,'low']   = np.log(dataframe.loc[:,'low'].pct_change()+1)
            dataframe.loc[:,'close'] = np.log(dataframe.loc[:,'close'].pct_change()+1)
            # dataframe.loc[:,'open_log']  = np.log(dataframe.loc[:,'open'].pct_change()+1)
            # dataframe.loc[:,'high_log']  = np.log(dataframe.loc[:,'high'].pct_change()+1)
            # dataframe.loc[:,'low_log']   = np.log(dataframe.loc[:,'low'].pct_change()+1)
            # dataframe.loc[:,'close_log'] = np.log(dataframe.loc[:,'close'].pct_change()+1)

        # Make the triggers values binary : -1/1
        # Doc : df.loc[<row selection>, <column selection>]
        dataframe.loc[dataframe.buys.isna(),  'buys']  = 0
        dataframe.loc[dataframe.buys != 0,    'buys']  = -1
        dataframe.loc[dataframe.sells.isna(), 'sells'] = 0
        dataframe.loc[dataframe.sells != 0,   'sells'] = +1

        if 'index' in dataframe.columns:
            del dataframe['index']

        # Set index
        dataframe.set_index('time', inplace=True)

        # Reformat datetime index, Binance's data is messy
        dataframe.index = pd.to_datetime(dataframe.index, format='%Y-%m-%d %H:%M:%S.%f')

        return dataframe.iloc[1:]


    def histograms_from_CSV(self)->dict:
        """ Lecture du .csv contenant les histogrames et transformation en Series. """

        hists_dict = dict()

        histos_file = f'histograms/{self.quote}/{self.timeframe}/{self.pair}_{self.timeframe}_{self.side}'
        for chunk in pd.read_csv(histos_file,
                                 sep       = '\t',
                                 index_col = 0,
                                 chunksize = 13,             # Choose nbins+1. The chunksize parameter specifies the number of rows per chunk. (The last chunk may contain fewer than chunksize rows, of course.).
                                 header    = None,
                                 ):

            histogram_name = chunk.iloc[0,0]

            # Convert the string to an Interal
            str_intervals = [i.replace("(","").replace("]", "").split(", ") for i in chunk.index[1:]]                       # Keep the 2 floats only
            original_cuts = [pd.Interval(float(i), float(j)) for i, j in str_intervals]                                     # Convert the 2 floats to an Interval
            # Create a Series from the dataframe, that has the Intervals as index.
            cleaned_values = [float(i.item().replace('[','').replace(']', '')) for i in chunk.values[1:]]					# I don't know why, the values are in single-element lists no we need to convert them to flaots
            histogram = pd.Series(data=cleaned_values, name=histogram_name, index=original_cuts)

            # Populate the histograms' dictionnary
            hists_dict[histogram_name] = histogram

        return hists_dict



if __name__ == '__main__':

    quote_     = 'BTC'
    pair_      = 'ETHBTC'
    timeframe_ = '5m'

    # In the project directory, create a nested directory for the quoteasset if not exists
    # Path('models/' + quote_).mkdir(parents=True, exist_ok=True)
    # Path('probas/' + quote_).mkdir(parents=True, exist_ok=True)

    print(f"_________ {pair_} _________")

    for side_ in ['sells']:
        # for side_ in ['buys', 'sells']:
        # print(f"{side_[:-1].capitalize()} side.")

        # for algo in ['LogisticRegression', 'RandomForest', 'DecisionTree', 'MLP', 'KNeighbors']:
        for algo in ['RandomForest']:
            print(algo)
            model = ModelSelection(quote=quote_, pair=pair_, timeframe=timeframe_, side=side_, log=True)
            model.fit_save__and_evaluate_model(sklearn_model=algo)                                          # ['LogisticRegression', 'RandomForest', 'DecisionTree', 'MLP', 'KNeighbors']
