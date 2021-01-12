import pandas as pd, csv
import pandas_ta as ta
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib  import Path
import os


# Test ___________________________________________________________________________________________
def Custom_Strategy(historical_df, plot:bool):
    """ Computes the correlation heatmap between all the indicators. """

    # if 'index' in historical_df.columns:
    #     del historical_df['index']

    historical_df.rename(columns={"open_log_returns": "open", "high_log_returns": "high", "low_log_returns": "low", "close_log_returns": "close"}, inplace=True)

    # Pandas-ta deletes the rows that contain NaNs, so we need to delete the buys and sells
    # del historical_df['buys']
    # del historical_df['sells']
    # historical_df.drop([0], inplace=True)

    # Create a custom strategy object containing the indicators to compute
    CustomStrategy = ta.Strategy(name="Custom Strat",
                                 ta=[
                                     {"kind": "kurtosis", "params": (50,)},
                                     {"kind": "macd",     "params": (25, 15, 9)},
                                 ],
                                 )

    # Run the CustomStrategy
    historical_df.ta.strategy(CustomStrategy)

    print(historical_df.tail())

    # Use verbose if you want to make sure it is running.
    # historical_df.ta.strategy(verbose=True)

    # Use timed if you want to see how long it takes to run.
    # historical_df.ta.strategy(timed=True)

    # Get the list of the least correlated indicators
    # uncorr_indicators = Corr_matrix(dataframe=historical_df, threshold=0.4, plot=plot, category='Custom')
# ___________________________________________________________________________________________


class Histograms:

    def __init__(self, quote, pair, timeframe):
        self.df     = self.get_df(quote=quote, pair=pair, timeframe=timeframe)
        self.initdf = self.df.copy()
        self.quote  = quote
        self.pair   = pair
        self.timeframe = timeframe
        self.unwantedinds_buys  = []
        self.unwantedinds_sells = []

        # Clear the csv files containing the histograms if they exist
        for side in ['_buys', '_sells']:
            with open(histos_file + side, 'w'):
                pass
            with open(histos_file + side  + '_temp', 'w'):
                pass

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
            # ssf         = (30,3),
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
        self.custom_inds_list = list(self.all_custom_inds.keys())   # Not used yet


    def add_indicators(self):

        print("\tComputing the indicators")

        ta_strat = [{'kind'     : ind,
                     'params'   : params,
                     'col_names': tuple(["{name}_{params}".format(name=f'{ind}{str(i)}', params='_'.join(list(map(str,params))) if len(params)>1 else params[0] if len(params)==1 else 'None') for i in range(5)])
                     }
                    for ind, params in self.all_custom_inds.items()]

        # Create a custom strategy object containing the indicators to compute
        CustomStrategy = ta.Strategy(name = "MyStrat",
                                     ta   = ta_strat)

        # Run the CustomStrategy on the df : compute the indicators
        self.df.ta.cores = 6
        self.df.ta.strategy(CustomStrategy,
                            timed = True,
                           )

        # self.df.to_csv(f'historical_data_with_inds/{self.quote}/{self.timeframe}/{self.pair}_{self.timeframe}', sep='\t', index=False, na_rep="NaN")

        # with pd.option_context('display.max_rows', 10, 'display.max_columns', None):
        #     print(self.df)


    def computeandsave_histos(self, nbins:int, plot_hists:bool):
        """ Computes the indicator column(s) and adds it to the dataframe. Plots if asked. """

        self.add_indicators()
        # self.df = pd.read_csv(f'historical_data_with_inds/{self.quote}/{self.timeframe}/{self.pair}_{self.timeframe}', sep='\t')

        all_inds = list(self.df.columns[7:])

        print(f'\tComputed {len(list(self.df.columns[7:]))} indicators')

        print("\tComputing the histograms")

        # For each column, compute probabilities and save histograms
        for i, col in enumerate(all_inds):
            try:
                df_ = self.df.copy()

                # Define colnames for buys and sells for this indicator
                ind_buys  = "{name}_{side}".format(name=col, side='buys')
                ind_sells = "{name}_{side}".format(name=col, side='sells')

                # Value of the indicator at the triggers
                df_.loc[:,ind_buys]  = np.where(self.df['buys']==-1,  df_[col], float('nan'))
                df_.loc[:,ind_sells] = np.where(self.df['sells']==+1, df_[col], float('nan'))

                # Compute the bins of the histogram
                # Ideal bins number : sqrt(total number of samples) = np.sqrt(len(df[kurtosis_name]))
                df_[col].replace(np.inf,np.nan,inplace=True)                        # Replaces inf by nan
                minVal    = df_[col].min()
                maxVal    = df_[col].max()
                lst       = list(np.linspace(minVal, maxVal, nbins+1))
                intervals = [tuple(lst[k:k+2]) for k in range(0, len(lst)-1, 1)]        # intervals = [(-5, -4.5), (-4.5, -4), (-4, -3.5), (-3.5, -3), (-3, -2.5), (-2.5, -2), (-2, -1.5), (-1.5, -1), (-1, -0.5), (-0.5, 0), (0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2.5), (2.5, 3), (3, 3.5), (3.5, 4), (4.5, 5), (5, 5.5)]
                bins      = pd.IntervalIndex.from_tuples(intervals)

                # Compute the histograms
                histogram_full  = pd.cut(df_[col].dropna(),       bins=bins).value_counts().sort_index()
                histogram_buys  = pd.cut(df_[ind_buys].dropna(),  bins=bins).value_counts().sort_index()
                histogram_sells = pd.cut(df_[ind_sells].dropna(), bins=bins).value_counts().sort_index()
                # Ratio of the 2 : if a point falls into a bar, its probabibility to be a buy point is equal to the bar's
                histogram_ratio_buys  = histogram_buys.div(histogram_full).fillna(0)          # pd.Series   # If a bin's value is NaN, replace it with 0.
                histogram_ratio_sells = histogram_sells.div(histogram_full).fillna(0)         # pd.Series   # If a bin's value is NaN, replace it with 0.

                # Format the labels of the plot to 2 decimal places, for readability :
                # https://stackoverflow.com/questions/62935231/matplotlib-specify-format-of-bin-values-in-a-histograms-tick-labels
                xtl = [f'({l:.2f}, {r:.2f}]' for l,r in zip(bins.values.left, bins.values.right)]

                if plot_hists:
                    plt.figure(figsize = (16,12))

                    # First line
                    plt.subplot(2, 3, 1)
                    plt.plot(df_[col], color='black')
                    plt.title(col + ' on log prices')
                    plt.ylabel(col)
                    plt.subplot(2, 3, 2)
                    histogram_buys.plot(kind='bar')
                    plt.title(col + " when it's a buy")
                    plt.xlabel(col + ' value')
                    plt.ylabel('Realisations')
                    plt.gca().set_xticklabels(xtl)
                    plt.subplot(2, 3, 3)
                    histogram_ratio_buys.plot(kind='bar', color='red')
                    plt.title('Probability that the ' + col + ' value indicates a buy')
                    plt.xlabel(col + ' value')
                    plt.ylabel('Probability')
                    plt.gca().set_xticklabels(xtl)

                    # Second line
                    plt.subplot(2, 3, 4)
                    histogram_full.plot(kind='bar', color='black')
                    plt.title('All ' + col + ' realisations')
                    plt.xlabel(col + ' value')
                    plt.ylabel('Realisations')
                    plt.gca().set_xticklabels(xtl)
                    plt.subplot(2, 3, 5)
                    histogram_sells.plot(kind='bar')
                    plt.title(col + " when it's a sell")
                    plt.xlabel(col + ' value')
                    plt.ylabel('Realisations')
                    plt.gca().set_xticklabels(xtl)
                    plt.subplot(2, 3, 6)
                    histogram_ratio_sells.plot(kind='bar', color='green')
                    plt.title('Probability that the ' + col + ' value indicates a sell')
                    plt.xlabel(col + ' value')
                    plt.ylabel('Probability')
                    plt.gca().set_xticklabels(xtl)
                    plt.show()

                for hist, side in zip([histogram_ratio_buys, histogram_ratio_sells], ['buys', 'sells']):
                    # print(hist)
                    # Save the histograms to the temporary CSV only if they don't have more than 50% of the bins != 0. If not, also delete the columns from self.df.
                    if hist.value_counts().to_dict().get(0.0, 0) < len(hist)/2:
                        hist.to_frame(name=col).to_csv(f'{histos_file}_{side}_temp',
                                                       sep    = '\t',
                                                       index  = True,
                                                       mode   = 'a',
                                                       )
                    else:
                        # Populate the list of ofwanted indicators
                        getattr(self, f'unwantedinds_{side}').append(col)

            except Exception as e:
                print(f"Exception occurred when trying to compute the histograms of {col}.")
                print(e)


    def findandsave_leastcorrelated(self, plot_tot_corr_matrix:bool=False):
        """ Compute all the indicators and keep the least correlated ones."""

        bins = 12

        self.computeandsave_histos(nbins=bins, plot_hists=False)

        for side in ['buys', 'sells']:
            print(f"\t{side[0:-1].title()} side")
            print(f"\t\tExcluded {len(getattr(self, f'unwantedinds_{side}'))} indicators that result in bad histograms: {getattr(self, f'unwantedinds_{side}')}.")
            # Compute the correlations on the indicators that don't result in strange histograms (that have more that 50% bins at 0)
            col_subset  = [x for x in list(self.df.columns)[7:] if x not in getattr(self, f'unwantedinds_{side}')]
            uncorr_inds = self.corr_matrix(dataframe=self.df.loc[:,col_subset], threshold=0.4, plot=plot_tot_corr_matrix)

            # Keep only the usefull histograms in the CSV
            self.delete_useless_histograms_from_CSV(file=f'{histos_file}_{side}_temp',  usefull_hists=uncorr_inds,  nbins=bins)

            # At this point, we have produced two csv files containing all the relevant histograms :)
            print(f"\t\tCorrelations : saved {len(uncorr_inds)} of {len(col_subset)} valid histograms, out of {len(list(self.df.columns)[7:])}.")


    # Utilities ______________________________________________________________________________________

    @staticmethod
    def get_df(quote:str, pair:str, timeframe:str):
        """ Gets the historic data from the csv file """

        # # Work on historical data from the csv file
        historical_data_file = f'../historical_data/{quote}/{timeframe}/{pair}_{timeframe}'
        dataframe_ = pd.read_csv(historical_data_file, sep='\t')

        # del dataframe_['time']
        # dataframe_.rename(columns={"open_log_returns": "open", "high_log_returns": "high", "low_log_returns": "low", "close_log_returns": "close"}, inplace=True)

        # Remove duplicated lines in the historical data if present
        dataframe = dataframe_.loc[~dataframe_.index.duplicated(keep='first')]

        # Compute the log returns
        dataframe.loc[:,'open']  = np.log(dataframe.loc[:,'open'].pct_change()+1)
        dataframe.loc[:,'high']  = np.log(dataframe.loc[:,'high'].pct_change()+1)
        dataframe.loc[:,'low']   = np.log(dataframe.loc[:,'low'].pct_change()+1)
        dataframe.loc[:,'close'] = np.log(dataframe.loc[:,'close'].pct_change()+1)
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

        return dataframe

    @staticmethod
    def corr_matrix(dataframe, threshold, plot:bool)->list:

        # Correlation of all the indicators
        corr_matrix_tot = dataframe.corr().abs()
        #  mask out the top triangle
        mask = np.zeros_like(corr_matrix_tot)
        mask[np.triu_indices_from(mask)] = True
        # Plot the correlation heatmap using Seaborn
        f = plt.figure(figsize = (16,12))
        plt.title('Correlation of all the {n} indicators'.format(n=len(list(corr_matrix_tot.columns))), fontsize=20)
        full_ax = sns.heatmap(corr_matrix_tot, cmap="Blues", vmin=-1, vmax=1, square=True, annot=True, mask=mask)


        def remove_correlated_columns(dataset, seuil):
            # Work on a copy of a dataframe, so that we don't change the actual one
            dataf = dataset.copy()

            col_corr = set()                                    # Set of all the names of deleted columns
            corr_matrix = dataf.corr().abs()                    # .abs() : Take the absolute correlation
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    if (corr_matrix.iloc[i, j] >= seuil) and (corr_matrix.columns[j] not in col_corr):
                        colname = corr_matrix.columns[i]        # getting the name of column
                        col_corr.add(colname)
                        if colname in dataf.columns:
                            del dataf[colname]                  # deleting the column from the dataset

            return dataf

        # Correlation of all the uncorrelated indicators
        df_uncorr_tot = remove_correlated_columns(dataset=dataframe, seuil=threshold)
        uncorr_corr_matrix = df_uncorr_tot.corr().abs()
        #  mask out the top triangle
        mask = np.zeros_like(uncorr_corr_matrix)
        mask[np.triu_indices_from(mask)] = True
        uncorr_f  = plt.figure(figsize = (16,12))
        plt.title('Least correlated indicators (|c| < {threshold})    ({n} inds)'.format(threshold=threshold, n=len(list(uncorr_corr_matrix.columns))), fontsize=20)
        uncorr_ax = sns.heatmap(uncorr_corr_matrix, cmap="Blues", vmin=0, vmax=threshold, square=True, annot=True, mask=mask)

        if plot:
            f.show()
            uncorr_f.show()

        # Return the list of the uncorrelated indicators
        uncorr_inds = list(uncorr_corr_matrix.columns)

        return uncorr_inds

    @staticmethod
    def delete_useless_histograms_from_CSV(file, usefull_hists:list, nbins:int):
        """ Creates a new csv files containing only the relevant histograms,
            and deletes the csv containing all the histograms"""

        def histograms_from_CSV(filename)->dict:
            """ Lecture du .csv contenant les histogrames et transformation en Series. """

            histograms_dict = dict()

            for chunk in pd.read_csv(filename,
                                     sep       = '\t',
                                     index_col = 0,
                                     chunksize = nbins+1,             # Choose nbins+1. The chunksize parameter specifies the number of rows per chunk. (The last chunk may contain fewer than chunksize rows, of course.).
                                     header    = None,
                                     ):

                histogram_name = chunk.iloc[0,0]

                # Convert the string to an Interal
                str_intervals = [i.replace("(","").replace("]", "").split(", ") for i in chunk.index[1:]]                       # Keep the 2 floats only
                original_cuts = [pd.Interval(float(i), float(j)) for i, j in str_intervals]                                     # Convert the 2 floats to an Interval
                # Create a Series from the dataframe, that has the Intervals as index.

                # for i in chunk.values[1:]:
                #     print(histogram_name, float(i.item().replace('[','').replace(']', '')))

                cleaned_values = [float(i.item().replace('[','').replace(']', '')) for i in chunk.values[1:]]					# I don't know why, the values are in single-element lists no we need to convert them to flaots
                histogram = pd.Series(data=cleaned_values, name=histogram_name, index=original_cuts)

                # Populate the histograms' dictionnary
                histograms_dict[histogram_name] = histogram

            return histograms_dict

        # Get a dict of all the histograms that are in the temp file
        hist_dict = histograms_from_CSV(file)

        # Delete the key, value pairs that we don't need
        keys = list(hist_dict.keys())
        for ind in keys:
            if ind not in usefull_hists:
                del hist_dict[ind]

        newfile = file.replace('_temp', '')
        with open(newfile, 'w', newline='') as f:
            # # Add a header to the new CSV : the indicators that are in it
            # w = csv.writer(f)
            # w.writerow(usefull_hists)
            # # Or just clear the file
            pass

        # Save the dict to the new CSV.
        with open(newfile, 'a'):
            for ind, hist in hist_dict.items():
                hist.to_frame(name=ind).rename_axis('bins').to_csv(newfile, sep='\t', index=True, na_rep='NaN', mode='a')

        # Delete the CSV that contains all the histograms, we don't need it anymore
        os.remove(file)




if __name__ == "__main__":

    quote_     = 'BTC'
    pair_      = 'ETHBTC'
    timeframe_ = '5m'

    print(f"_________ {pair_} _________")

    # In the project directory, create a nested directory for the quoteasset if not exists
    Path( f'histograms/{quote_}/{timeframe_}').mkdir(parents=True, exist_ok=True)
    # Path('historical_data_with_inds/' + quote_).mkdir(parents=True, exist_ok=True)

    # Common name for the csv files that will contain the buy & sell histograms
    histos_file = f'histograms/{quote_}/{timeframe_}/{pair_}_{timeframe_}'

    histograms = Histograms(quote=quote_, pair=pair_, timeframe=timeframe_)
    histograms.findandsave_leastcorrelated(plot_tot_corr_matrix=False)