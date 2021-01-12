from datetime import datetime
import pandas as pd
import numpy as np
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
import matplotlib.pyplot as plt
import itertools
import pandas_ta as ta
from empyrical import sortino_ratio, calmar_ratio, omega_ratio, sharpe_ratio


def ADCF_test(pairs:tuple):

    print(pairs)
    pair0 = pairs[0]
    pair1 = pairs[1]

    # max_ = None
    max_ = 2000

    # Get the dataframes from the csv files, keep only the time and close columns
    df_pair0 = pd.read_csv(f'historical_data/{"BTC"}/{pair0}_{"2h"}', sep='\t').loc[:,['time', 'close']].loc[:max_]
    df_pair1 = pd.read_csv(f'historical_data/{"BTC"}/{pair1}_{"2h"}', sep='\t').loc[:,['time', 'close']].loc[:max_]
    # Rename the close columns to the pair's name
    df_pair0.columns = ['time', pair0]
    df_pair1.columns = ['time', pair1]
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

    # if pairs == ('LTCBTC', 'NEOBTC'):
    #     print(finaldf)

    # If the two coins don't have enough overlapping, skip the pair.
    if len(finaldf) < 1000:
        print(f'{pairs} - Less than 1000 ({len(finaldf)}) matching indexes, skipping the pair.')
        return

    # run Odinary Least Squares regression to find hedge ratio and then create spread series
    est = sm.OLS(finaldf.loc[:,pair0], finaldf.loc[:,pair1])
    est = est.fit()
    finaldf['hr'] = -est.params[0]
    # Create spread series
    finaldf.loc[:,'spread'] = finaldf.loc[:,pair1] + finaldf.hr*finaldf.loc[:,pair0]

    # Plot the spread
    # plt.plot(finaldf.spread)
    # plt.show()

    # Augmented Dickey Fuller test __________________________________________________________________
    cadf = ts.adfuller(finaldf.spread)
    print('Augmented Dickey Fuller test statistic =', cadf[0])
    print('Augmented Dickey Fuller p-value =', cadf[1])
    print('Augmented Dickey Fuller 1%, 5% and 10% test statistics =', cadf[4])

    if cadf[1] > 0.05:
        return

    cointegrated_pairs.append(pairs)

    # Half life of mean reversion __________________________________________________________________
    # Run OLS regression on spread series and lagged version of itself
    spread_lag         = finaldf.spread.shift(1)
    spread_lag.iloc[0] = spread_lag.iloc[1]
    spread_ret         = finaldf.spread - spread_lag
    spread_ret.iloc[0] = spread_ret.iloc[1]
    spread_lag2        = sm.add_constant(spread_lag)

    model = sm.OLS(spread_ret,spread_lag2)
    res = model.fit()

    # Half-life of the mean-reversion
    halflife = round(-np.log(2) / res.params[1],0)
    print('Half-life = ', halflife)


    # Backtest ______________________________________________________________________________________
    finaldf.ta.zscore(close=finaldf.loc[:,'spread'], length=halflife, std=1, append=True, col_names=('zScore',))

    entryZscore = 2
    exitZscore = 0

    # set up num units long
    finaldf['long entry'] = ((finaldf.zScore < - entryZscore) & (finaldf.zScore.shift(1) > - entryZscore))
    finaldf['long exit'] = ((finaldf.zScore > - exitZscore) & (finaldf.zScore.shift(1) < - exitZscore))
    finaldf['num units long'] = np.nan
    finaldf.loc[finaldf['long entry'], 'num units long'] = 1
    finaldf.loc[finaldf['long exit'],  'num units long'] = 0
    finaldf['num units long'][0] = 0
    finaldf['num units long'] = finaldf['num units long'].fillna(method='pad')  # where we need to be long
    # set up num units short
    finaldf['short entry'] = ((finaldf.zScore > entryZscore) & (finaldf.zScore.shift(1) < entryZscore))
    finaldf['short exit'] = ((finaldf.zScore < exitZscore) & (finaldf.zScore.shift(1) > exitZscore))
    finaldf.loc[finaldf['short entry'],'num units short'] = -1
    finaldf.loc[finaldf['short exit'],'num units short'] = 0
    finaldf['num units short'][0] = 0
    finaldf['num units short'] = finaldf['num units short'].fillna(method='pad')

    finaldf['numUnits'] = finaldf['num units long'] + finaldf['num units short']
    finaldf['spread pct ch'] = (finaldf['spread'] - finaldf['spread'].shift(1)) / ((finaldf[pair0] * abs(finaldf['hr'])) + finaldf[pair1])  # percentage change of the spread
    finaldf['port rets'] = finaldf['spread pct ch'] * finaldf['numUnits'].shift(1)      # portfolio return

    finaldf['cum rets'] = finaldf['port rets'].cumsum()
    finaldf['cum rets'] = finaldf['cum rets'] + 1

    plt.plot(finaldf.index, finaldf['cum rets'])
    plt.show()

    metric  = sharpe_ratio(finaldf['port rets'], annualization=1)
    print("Sharpe ratio = ", metric)



if __name__ == '__main__':

    pairs_to_test = ["ETHBTC", "LTCBTC", "LINKBTC", "UNIBTC", "BNBBTC", "DOTBTC", "YFIBTC", "NEOBTC", "XRPBTC"]
    pairs_combinations = list(itertools.combinations(pairs_to_test, 2))

    cointegrated_pairs = []

    # Run an Augmented Dickey Fuller test on all the combinations
    for comb in pairs_combinations:
        ADCF_test(pairs=comb)

    print(cointegrated_pairs)