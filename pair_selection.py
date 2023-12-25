import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta


def fetch_currency_data(currency_pairs, start_date, end_date):
    currency_pairs_tickers = [pair + "=X" for pair in currency_pairs]
    return yf.download(currency_pairs_tickers, start=start_date, end=end_date)['Adj Close']


def get_currency_pairs():
    return [
        "AUDUSD", "EURUSD", "NZDUSD", "GBPUSD", "BRLUSD", "CADUSD",
        "CNYUSD", "HKDUSD", "INRUSD", "KRWUSD", "MXNUSD", "ZARUSD",
        "SGDUSD", "DKKUSD", "JPYUSD", "MYRUSD", "NOKUSD", "SEKUSD",
        "LKRUSD", "CHFUSD", "TWDUSD", "THBUSD"
    ]


def pair_regression_analysis(df):
    regressions = {}
    # Generate all unique pair combinations for regression
    currency_pairs = df.columns  # Assuming these are your currency pair tickers
    for i, pair1 in enumerate(currency_pairs):
        for pair2 in currency_pairs[i + 1:]:  # Avoid self-pairing and repeats
            # get the adj close values for each currency pair
            pair1_values = df[pair1]
            pair2_values = df[pair2]

            # run ols
            regression = get_ols_metrics(pair1_values, pair2_values)

            # add the regression to the dictionary with correct tuple key
            regressions[(pair1, pair2)] = regression

    regressions_df = pd.concat(regressions).reset_index()
    regressions_df = regressions_df.rename(
        columns={'level_0': 'Currency Pair 1', 'level_1': 'Currency Pair 2'})
    return regressions_df.iloc[:, [0, 1, 3, 5, 6, 7]]


def get_ols_metrics(regressors, targets, annualization=1, ignorenan=True):
    # ensure regressors and targets are pandas dataframes, as expected
    if not isinstance(regressors, pd.DataFrame):
        regressors = regressors.to_frame()
    if not isinstance(targets, pd.DataFrame):
        targets = targets.to_frame()

    # align the targets and regressors on the same dates
    df_aligned = targets.join(regressors, how='inner', lsuffix='y ')
    Y = df_aligned[targets.columns]
    Xset = df_aligned[regressors.columns]

    reg = pd.DataFrame(index=targets.columns)
    for col in Y.columns:
        y = Y[col]

        if ignorenan:
            # ensure we use only non-NaN dates
            alldata = Xset.join(y, lsuffix='X')
            mask = alldata.notnull().all(axis=1)
            y = y[mask]
            X = Xset[mask]
        else:
            X = Xset

        model = LinearRegression().fit(X, y)
        reg.loc[col, 'alpha'] = model.intercept_ * annualization
        reg.loc[col, regressors.columns] = model.coef_
        reg.loc[col, 'r-squared'] = model.score(X, y)

        # sklearn does not return the residuals, so we need to build them
        yfit = model.predict(X)
        residuals = y - yfit

        # Treynor Ratio is only defined for univariate regression
        if Xset.shape[1] == 1:
            reg.loc[col, 'Treynor Ratio'] = (
                y.mean() / model.coef_) * annualization

        # if intercept =0, numerical roundoff will nonetheless show nonzero Info Ratio
        num_roundoff = 1e-12
        if np.abs(model.intercept_) < num_roundoff:
            reg.loc[col, 'Info Ratio'] = None
        else:
            reg.loc[col, 'Info Ratio'] = (
                model.intercept_ / residuals.std()) * np.sqrt(annualization)

    return reg


def select_high_alpha(data, alpha_threshold):
    """Select currency pairs with alpha greater than the specified threshold."""
    filtered_data = data[data['alpha'] > alpha_threshold]
    high_alpha = []
    for _, row in filtered_data.iterrows():
        high_alpha.append((row['Currency Pair 1'], row['Currency Pair 2']))
    return high_alpha


def select_high_treynor(data, treynor_threshold):
    """Select currency pairs with a Treynor Ratio greater than the specified threshold."""
    filtered_data = data[data['Treynor Ratio'] > treynor_threshold]
    high_treynor = []
    for _, row in filtered_data.iterrows():
        high_treynor.append((row['Currency Pair 1'], row['Currency Pair 2']))
    return high_treynor


def select_high_info_ratio(data, info_threshold):
    """Select currency pairs with an Information Ratio greater than the specified threshold."""
    filtered_data = data[data['Info Ratio'] > info_threshold]
    high_info_ratio = []
    for _, row in filtered_data.iterrows():
        high_info_ratio.append(
            (row['Currency Pair 1'], row['Currency Pair 2']))
    return high_info_ratio


def select_high_r_squared(data, r_squared_threshold):
    """Select currency pairs with an r-squared value greater than the specified threshold."""
    filtered_data = data[data['r-squared'] > r_squared_threshold]
    high_r_squared = []
    for _, row in filtered_data.iterrows():
        high_r_squared.append((row['Currency Pair 1'], row['Currency Pair 2']))
    return high_r_squared


def combine_filter_pairs(regressions_df):
    high_alpha_pairs = select_high_alpha(
        regressions_df,
        regressions_df['alpha'].mean() + regressions_df['alpha'].std())
    high_treynor_pairs = select_high_treynor(
        regressions_df,
        regressions_df['Treynor Ratio'].mean() + regressions_df['Treynor Ratio'].std())
    high_info_ratio_pairs = select_high_info_ratio(
        regressions_df,
        regressions_df['Info Ratio'].mean() + regressions_df['Info Ratio'].std())
    high_r_squared_pairs = select_high_r_squared(
        regressions_df,
        regressions_df['r-squared'].mean() + regressions_df['r-squared'].std())

    all_pairs = []
    for pair in high_alpha_pairs + high_treynor_pairs + high_info_ratio_pairs + high_r_squared_pairs:
        all_pairs.append(sorted(pair))

    pair_counts = {}
    # Iterate over all_pairs to count each occurrence
    for pair in all_pairs:
        # Convert the list pair to a tuple for use as a dictionary key
        tuple_pair = tuple(pair)  # convert list to tuple
        if tuple_pair in pair_counts:
            pair_counts[tuple_pair] += 1
        else:
            pair_counts[tuple_pair] = 1

    # Filter pairs that meet more than two criteria
    common_pairs = [pair for pair, count in pair_counts.items() if count >= 2]

    return common_pairs


def main():
    currency_pairs = get_currency_pairs()
    start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')

    df = fetch_currency_data(currency_pairs, start_date, end_date)
    regressions_df = pair_regression_analysis(df)
    common_pairs = combine_filter_pairs(regressions_df)

    print("Common Pairs:", common_pairs)


if __name__ == "__main__":
    main()
