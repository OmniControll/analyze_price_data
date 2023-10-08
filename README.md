# analyze_price_data

consists of 2 files: graph.stx and stock_analysis

graph.stx:

The primary objective of this project was to understand and analyze the behavior of Bitcoin's closing prices over time. 
This includes capturing the underlying patterns, trends, and seasonality inherent in the data, aiming to provide an informed basis for financial decision-making or predictive modeling.

stock_analysis:

The objective of this project is to provide a comprehensive tool for stock portfolio analysis and optimization. The tool uses Monte Carlo simulation to model the behavior of a portfolio consisting of various stocks. 
it employs Sharpe Ratio maximization to find the most efficient asset allocation. 
Finally, it calculates the downside deviation for assessing portfolio risk and visualizes the efficient frontier using Plotly.

calculations:

Financial Calculations: Calculated daily returns, expected returns, and the covariance matrix for the given stock data.

Monte Carlo Simulation: Simulated thousands of portfolios with random asset allocations to calculate portfolio returns, volatilities, and Sharpe ratios.

Sharpe Ratio Optimization: Employed the SciPy library to minimize the negative of the Sharpe ratio, thus finding the asset allocation that maximizes it.

Downside Deviation: Calculated the downside deviation of the portfolio returns to assess the risk more accurately.

Requirements

Python 3.x
Libraries: numpy, pandas, scipy, yfinance, plotly

