import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from pair_selection import get_currency_pairs, fetch_currency_data, pair_regression_analysis, combine_filter_pairs
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta


# Function to calculate the Z-score
def zscore(series):
    return (series - series.mean()) / np.std(series)


def exit_logic(pair, data, entry_price, exit_threshold, stop_loss_percent, take_profit_percent, max_holding_period):
    # Initialize variables
    stop_loss_price = entry_price * (1 - stop_loss_percent)
    take_profit_price = entry_price * (1 + take_profit_percent)
    holding_period = 0

    for index, row in data.iterrows():
        current_price = row[pair[0]]  # Get current price of the asset
        # Assuming you have calculated z-score for the spread and included it in the data
        spread_zscore = row['spread_zscore']

        # Check for stop-loss or take-profit triggers
        if current_price <= stop_loss_price or current_price >= take_profit_price:
            return {'exit_signal': 'stop_take_profit', 'exit_price': current_price}

        # Check for mean reversion or z-score exit threshold
        if abs(spread_zscore) < exit_threshold:
            return {'exit_signal': 'zscore_reversion', 'exit_price': current_price}

        # Check for time-based exit
        holding_period += 1
        if holding_period >= max_holding_period:
            return {'exit_signal': 'time_based', 'exit_price': current_price}

    # If no exit condition met (data ends), return holding
    return {'exit_signal': 'holding', 'exit_price': current_price}


def backtest_pairs_trading_strategy(common_pairs, df, lookback=1000, entry_threshold=1, exit_threshold=1, stop_loss_percent=0.05, take_profit_percent=0.05, max_holding_period=30):
    # Initialize portfolio with no open positions
    portfolio = pd.DataFrame(index=df.index, data={'positions': [
                             0] * len(df.index)}, dtype=float)

    # Initialize account with starting capital
    account = pd.DataFrame(index=df.index, data={'balance': [
                           100000] * len(df.index)}, dtype=float)

    for pair in common_pairs:
        # Fetch historical data for the pair
        data = df[[pair[0], pair[1]]]

        # Calculate the spread and its Z-score
        spread = data[pair[0]] / data[pair[1]]
        spread_mean = spread.rolling(window=lookback, min_periods=1).mean()
        spread_std = spread.rolling(window=lookback, min_periods=1).std()
        data['spread_zscore'] = (spread - spread_mean) / spread_std

        for date, row in data.iterrows():
            current_price_0 = row[pair[0]]
            current_price_1 = row[pair[1]]
            current_zscore = row['spread_zscore']

            # Entry logic
            if current_zscore > entry_threshold and portfolio.at[date, 'positions'] == 0:
                # Entering trade
                print(
                    f"Entering trade on {date}: LONG {pair[0]}, SHORT {pair[1]}")
                portfolio.at[date, 'positions'] = 1  # Long the pair
                entry_price = current_price_0

            elif current_zscore < -entry_threshold and portfolio.at[date, 'positions'] == 0:
                # Entering trade
                print(
                    f"Entering trade on {date}: LONG {pair[1]}, SHORT {pair[0]}")
                portfolio.at[date, 'positions'] = -1  # Short the pair
                entry_price = current_price_1

            # Implement improved exit logic
            if portfolio.at[date, 'positions'] != 0:
                exit_info = exit_logic(
                    pair, data.loc[:date], entry_price, exit_threshold, stop_loss_percent, take_profit_percent, max_holding_period)

                # Exit logic based on improved exit strategy
                if exit_info['exit_signal'] != 'holding':
                    # Closing trade
                    print(
                        f"Exiting trade on {date} due to {exit_info['exit_signal']}")
                    exit_price = exit_info['exit_price']
                    # Calculate P&L from this trade
                    # Example position size
                    pnl_change = (exit_price - entry_price) * \
                        portfolio.at[date, 'positions'] * 1000
                    account.at[date, 'balance'] += pnl_change
                    portfolio.at[date, 'positions'] = 0  # Close position

    # Calculate final P&L for the entire period
    account['pnl'] = account['balance'] - account['balance'].iloc[0]

    # Combine portfolio, account, and pnl into a single DataFrame
    results = pd.concat([portfolio, account], axis=1).dropna()

    # Print final result
    print(f"Final account balance: ${results['balance'].iloc[-1]}")
    print(f"Net profit: ${results['pnl'].iloc[-1]}")

    # Plot the results
    results[['balance', 'pnl']].plot(
        subplots=True, title='Pairs Trading Strategy PnL and Balance')
    plt.show()

    return results


def maximumDrawdown(data):
    # Convert data to cumulative returns
    cum_returns = (1 + data).cumprod()
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    end_date = drawdown.idxmin()

    # Create a summary DataFrame
    summary = pd.DataFrame({
        'Max Drawdown': max_drawdown,
        'Trough Date': end_date,
        'Peak Date': rolling_max.idxmax(),
        'Recovery Date': drawdown[drawdown >= 0].idxmin()
    })

    summary['Duration (to Recover)'] = summary['Recovery Date'] - \
        summary['Peak Date']

    return summary


def tailMetrics(data, quantile=0.01, relative=False):
    metrics = pd.DataFrame(index=data.columns)
    metrics['Skewness'] = data.skew()
    metrics['Kurtosis'] = data.kurtosis()

    # Calculate VaR and CVaR
    VaR = data.quantile(quantile)
    CVaR = data[data <= VaR].mean()

    if relative:
        mean = data.mean()
        std = data.std()
        VaR = (VaR - mean) / std
        CVaR = (CVaR - mean) / std

    metrics[f'VaR ({quantile})'] = VaR
    metrics[f'CVaR ({quantile})'] = CVaR

    # Maximum Drawdown statistics
    mdd_stats = maximumDrawdown(data)
    metrics = metrics.join(mdd_stats)

    return metrics


def main():
    currency_pairs = get_currency_pairs()
    start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')

    df = fetch_currency_data(currency_pairs, start_date, end_date)
    regressions_df = pair_regression_analysis(df)
    common_pairs = combine_filter_pairs(regressions_df)
    results = backtest_pairs_trading_strategy(common_pairs, df)

    # Calculate maximum drawdown and tail metrics
    pnl_data = results['pnl']
    drawdowns = maximumDrawdown(pnl_data)
    tail_metrics = tailMetrics(pnl_data)

    print("Maximum Drawdowns:")
    print(drawdowns)
    print("\nTail Metrics:")
    print(tail_metrics)


if __name__ == "__main__":
    main()
