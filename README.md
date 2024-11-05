# Quant-Finance-Classes
Classes to implement common quantitative methods in Finance

Includes:

1. `asset.py`: Class to download (from `yfinance`) and process asset price data.
    Currently available methods include:
    
    - `to_returns()`: to obtain simple returns and log-returns from the price data
    - `summary_stats()`: descriptive statistics of log-returns including skewness and kurtosis
    - `get_allocations()`: weight allocations for maximum sharpe ratio portfolio from the historical data asset returns
    - `get_portfolio_returns()`: for a given asset allocations to get historical returns of the hypothetical portfolio
    - `mc_sim()`: performs Monte Carlo simulation of the portfolio value and returns the latter at the last time step. Have the ability to optionally plot the different portfolio paths
   
3. `risk.py`: Class to implement standard risk assessment methods using quantitative measures such as value at risk (VaR) and conditional value at risk (CVaR) in finance. 

    Currently available methods include:

    - VaR and CVaR estimates within the historical method
    - VaR and CVaR estimates using Monte Carlo method
