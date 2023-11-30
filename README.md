# analyze_price_data

consists of 2 files: graph.stx and stock_analysis

graph.stx:

The primary objective of this project was to understand and analyze the behavior of Bitcoin's closing prices over time. 
This includes capturing the underlying patterns, trends, and seasonality inherent in the data, aiming to provide an informed basis for financial decision-making or predictive modeling.

stock_analysis:

Portfolio Optimization Project

Overview

This project involves the development of a portfolio optimization tool using Python. It leverages Monte Carlo simulation to explore a vast space of portfolio allocations, aiming to maximize the Sharpe ratio, which is a measure of risk-adjusted return. The tool fetches real-time financial data, performs statistical calculations, and uses advanced optimization techniques to suggest the most efficient portfolio allocation.

Key Features

	1.	Data Fetching: Utilizes yfinance to fetch historical stock data.
	2.	Statistical Analysis: Calculates daily returns, expected returns, and the covariance matrix of the selected stocks.
	3.	Risk Metrics Calculation: Computes various risk metrics such as RSI, MACD, maximum drawdown, and Value at Risk (VaR).
	4.	Monte Carlo Simulation: Generates thousands of random portfolio allocations to visualize the risk-return profile of potential portfolios.
	5.	Sharpe Ratio Optimization: Uses the scipy.optimize module to find the portfolio allocation that maximizes the Sharpe ratio.
	6.	Interactive Dashboard: Implemented with Dash, allowing users to input stock tickers, view results, and interact with the data visualization.

Technologies Used

	•	Python
	•	Pandas for data manipulation
	•	yfinance for data fetching
	•	NumPy for numerical calculations
	•	Plotly and Dash for data visualization and web app development
	•	SciPy for optimization routines

Project Structure

	1.	Data Collection Module: Handles fetching and initial processing of stock data.
	2.	Financial Calculations Module: Computes daily returns, RSI, MACD, and other financial indicators.
	3.	Optimization Module: Contains functions for portfolio optimization, including Monte Carlo simulation and Sharpe ratio maximization.
	4.	Visualization Module: Uses Plotly to create interactive charts and graphs.
	5.	Dashboard Application: A Dash-based web app providing a user interface for the tool.

Usage

The user inputs a list of stock tickers into the Dash interface. The system fetches data for these stocks, performs analyses, and presents the optimized portfolio along with visualizations of potential portfolios based on the Monte Carlo simulation.

Future Enhancements

	•	Implementing more sophisticated risk models.
	•	Integrating real-time data for live portfolio recommendations.
	•	Expanding the range of financial instruments beyond stocks (e.g., bonds, ETFs).

Note

This tool is for informational purposes and should not be considered as financial advice. Users should perform their due diligence before making investment decisions.


