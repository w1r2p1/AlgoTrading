# check cointegration of pairs
from Exchange import Binance
from Database import BotDatabase

from statsmodels.tsa.vector_ar.vecm import coint_johansen
import numpy as np
import pandas as pd
import itertools
from datetime import datetime
from tqdm import tqdm


cointegrated_pairs = []


# pair is a df containing two ETF close price series
def check_johansen(comb, cointegrating_pairs, timeframe):

    COINTEGRATION_CONFIDENCE_LEVEL = 90 # require cointegration at 90%, 95%, or 99% confidence

    max_ = 5000
    # Get the dataframes from the csv files, keep only the time and close columns
    df_pair0 = pd.read_csv(f'historical_data/{"BTC"}/{timeframe}/{comb[0]}_{timeframe}', sep='\t').loc[:,['time', 'close']].loc[:max_]
    df_pair1 = pd.read_csv(f'historical_data/{"BTC"}/{timeframe}/{comb[1]}_{timeframe}', sep='\t').loc[:,['time', 'close']].loc[:max_]
    # Rename the close columns to the pair's name
    df_pair0.columns = ['time', comb[0]]
    df_pair1.columns = ['time', comb[1]]
    # Set indexes
    df_pair0.set_index('time', inplace=True)
    df_pair1.set_index('time', inplace=True)

    # Remove duplicated lines in the historical data
    df0 = df_pair0.loc[~df_pair0.index.duplicated(keep='first')]
    df1 = df_pair1.loc[~df_pair1.index.duplicated(keep='first')]

    # Reformat datetime index, Binance's data is messy
    df0.index = pd.to_datetime(df0.index, format='%Y-%m-%d %H:%M:%S.%f')
    df1.index = pd.to_datetime(df1.index, format='%Y-%m-%d %H:%M:%S.%f')

    # Merge in a single dataframe that has the time as index
    finaldf = pd.merge(df0, df1, how='outer', left_index=True, right_index=True)

    # Drop NaNs
    finaldf.dropna(0, inplace=True)

    # If the two coins don't have enough overlapping, skip the pair.
    if len(finaldf) < 1000:
        # print(f'{comb} - Less than 1000 ({len(finaldf)}) matching indexes, skipping the pair.')
        return

    # The second and third parameters indicate constant term, with a lag of 1.
    # https://www.statsmodels.org/stable/generated/statsmodels.tsa.vector_ar.vecm.JohansenTestResult.html#statsmodels.tsa.vector_ar.vecm.JohansenTestResult
    result = coint_johansen(finaldf, 0, 1)

    # the 90%, 95%, and 99% confidence levels for the trace statistic and maximum
    # eigenvalue statistic are stored in the first, second, and third column of
    # cvt and cvm, respectively
    confidence_level_cols = {90: 0, 95: 1, 99: 2}
    confidence_level_col = confidence_level_cols[COINTEGRATION_CONFIDENCE_LEVEL]

    trace_crit_value = result.cvt[:, confidence_level_col]      # Critical value (90%, 95%, 99%) of trace statistic.
    eigen_crit_value = result.cvm[:, confidence_level_col]      # Critical value (90%, 95%, 99%) of maximum eigenvalue statistic.
    # print("trace_crit_value = ", trace_crit_value)
    # print("eigen_crit_value = ", eigen_crit_value)
    # print("lr1 = ", result.lr1)                                 # Trace statistic
    # print("lr2 = ", result.lr2)                                 # Maximum eigenvalue statistic
    # The trace statistic and maximum eigenvalue statistic are stored in lr1 and lr2;
    # see if they exceeded the confidence threshold
    if np.all(result.lr1 >= trace_crit_value) and np.all(result.lr2 >= eigen_crit_value):
        # print("The two datasets "+comb[0]+" and "+comb[1]+" are cointegrated")

        # The first i.e. leftmost column of eigenvectors matrix, result.evec, contains the best weights.
        v1 = result.evec[:,0:1]
        hr = v1/-v1[1]              # to get the hedge ratio divide the best_eigenvector by the negative of the second component of best_eigenvector
        # the regression will be: close of symbList[1] = hr[0]*close of symbList[0] + error
        # where the beta of the regression is hr[0], also known as the hedge ratio, and
        # the error of the regression is the mean reverting residual signal that you need to predict, it is also known as the "spread"
        # the spread = close of symbList[1] - hr[0]*close of symbList[0] or alternatively (the same thing):
        # do a regression with close of symbList[0] as x and lose of symbList[1] as y, and take the residuals of the regression to be the spread.
        cointegrating_pairs.append(dict(
            asset_1=comb[0],
            asset_2=comb[1],
            hedge_ratio=hr[0][0]
        ))

    return 0


if __name__ == '__main__':

    # database  = BotDatabase("database.db")
    # ExistingQuoteassets = list(set([dict(bot)['quoteasset'] for bot in database.GetAllBots()]))       # ['ETH', 'BTC']
    exchange  = Binance(filename='credentials.txt')
    pairs     = exchange.GetNameOfPairs_WithQuoteasset(['BTC'])                                         # {'ETH': ['QTUMETH', 'EOSETH',..], 'BTC':[]}

    pairs_combinations = list(itertools.combinations(pairs['BTC'], 2))

    for comb_ in tqdm(pairs_combinations):
        try:
            check_johansen(list(comb_), cointegrated_pairs, '1d')
        except Exception:
            continue

    df = pd.DataFrame(cointegrated_pairs)
    print('\ncointegrated_pairs = \n', df)

    df.sort_values(by=['asset_1', 'asset_2'], inplace=True)
    df.to_csv("cointegrated_BTC_pairs_1d.csv", sep='\t', index=False)

