# Currency Pair Selection and Trading Strategy

## Overview

This project is a comprehensive approach to currency pairs trading using Python. It consists of two primary modules: `pairs_selection.py` and `pairs_trading.py`, focusing on identifying currency pairs with potential for statistical arbitrage and executing a pairs trading strategy respectively.

### pairs_selection.py

This module is dedicated to the selection of currency pairs based on various financial metrics. It fetches historical forex data, calculates correlation matrices, and runs regression analyses to identify pairs with high potential for trading. It features several functions each serving a specific part of the currency pair selection process:

- **fetch_currency_data**: Downloads historical currency pairs data.
- **get_currency_pairs**: Lists all available currency pairs for analysis.
- **get_ols_metrics**: Performs Ordinary Least Squares (OLS) regression and calculates various trading metrics.
- **pair_regression_analysis**: Analyzes pairs using correlation and regression.
- **select_high_alpha/treynor/info_ratio/r_squared**: Filters pairs based on specified financial metrics.
- **combine_filter_pairs**: Combines various filters to identify the most promising currency pairs.
- **main**: Runs the entire pair selection process using the above functions.

### pairs_trading.py

Once the pairs are selected, this module implements a pairs trading strategy backtesting environment. It includes functions for strategy execution, including trade entry and exit logic, risk management, and performance visualization:

- **zscore**: Calculates the Z-score of a series, typically used for normalization.
- **exit_logic**: Determines when to exit a trade based on various criteria.
- **backtest_pairs_trading_strategy**: Backtests the pairs trading strategy using historical data.
- **main**: Runs the whole trading strategy from data fetching to backtesting.

### Installation and Setup

To run these scripts, you need Python installed on your machine along with the following packages:

- numpy
- pandas
- yfinance
- seaborn
- matplotlib
- sklearn

You can install these packages via pip:

```bash
pip install numpy pandas yfinance seaborn matplotlib sklearn
```

### Usage

1. **Setting Up the Environment**: Ensure all dependencies are installed.
2. **Running Pair Selection**: Execute `pairs_selection.py` to select currency pairs based on historical data and statistical metrics.
3. **Executing Pair Trading Strategy**: Run `pairs_trading.py` to backtest the pairs trading strategy on the selected currency pairs.

### Contributing

Contributions to this project are welcome! You can contribute in several ways:

- **Improving Code**: Refactor or add new features.
- **Strategy Enhancement**: Propose or implement enhancements to the trading strategy.
- **Documentation**: Improve or add to the documentation.

