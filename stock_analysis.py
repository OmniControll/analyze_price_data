import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import os

#This code is modular, with functions handling specific tasks.
#The goal is to handle data fetching, financial calculations, optimization, and visualization.
#The main function will ensure the sequence of operations
#first lets get the stock data from yfinance and do some basic financial calculations

def fetch_stock_data(tickers, start_date, end_date):
    stock_data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return stock_data

def calculate_daily_returns(stock_data):
    daily_returns = stock_data.pct_change().dropna()
    return daily_returns

def calculate_expected_returns(daily_returns):
    return daily_returns.mean()

def calculate_covariance_matrix(daily_returns):
    return daily_returns.cov()

def calculate_portfolio_variance(weights, covariance_matrix):
    return np.dot(weights.T, np.dot(covariance_matrix, weights))
    
#let's start with constructing our monte carlo simulation 

def monte_carlo_simulation(expected_returns, covariance_matrix, num_portfolios, risk_free_rate):
    num_assets = len(expected_returns)
    results = np.zeros((num_portfolios, num_assets + 3))  # +3 for return, volatility, and Sharpe ratio

    for i in range(num_portfolios):
        # random weights for our portfolio simulation
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)

        # Calculations (formulas for) portfolio return, volatility, and Sharpe ratio
        portfolio_return = np.sum(weights * expected_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

        # Let's store the results now
        results[i, 0:num_assets] = weights
        results[i, num_assets] = portfolio_return
        results[i, num_assets + 1] = portfolio_volatility
        results[i, num_assets + 2] = sharpe_ratio

    # Create a Df
    columns = [f'weight_{asset}' for asset in expected_returns.index] + ['return', 'volatility', 'sharpe_ratio']
    results_df = pd.DataFrame(results, columns=columns)

    return results_df

def optimize_sharpe_ratio(expected_returns, covariance_matrix, risk_free_rate):
    num_assets = len(expected_returns)
# to calculate the optimized (maximized) sharpe ratio, we need to minimize the negative of it. the function below calculates the var. and return, then outputs the negated sharpe ratio
    def objective_function(weights):
        portfolio_variance = calculate_portfolio_variance(weights, covariance_matrix)
        expected_portfolio_return = np.sum(weights * expected_returns)
        neg_sharpe_ratio = - (expected_portfolio_return - risk_free_rate) / np.sqrt(portfolio_variance)
        return neg_sharpe_ratio
# we want only one constraint: the sum of the weights must equal 100%. we use a lambda with x as argument to represent the weights of our assets in the portfolio
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    #bounds for each weight (0,1)
    bounds = [(0, 1) for _ in range(num_assets)]
    #starting point
    initial_weights = np.array([1 / num_assets] * num_assets)
    optimized_weights = minimize(objective_function, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return optimized_weights.x


# Now lets run our functions to analyze and optimize our stock portfolio. we'll use a risk free rate of 2%.
# We also list the simulated portfolios by their Sharpe ratio in descending order, and select the top 5..

def analyze_stocks(tickers, start_date, end_date, num_portfolios, risk_free_rate):
    stock_data = fetch_stock_data(tickers, start_date, end_date)
    daily_returns = calculate_daily_returns(stock_data)
    expected_returns = calculate_expected_returns(daily_returns)
    covariance_matrix = calculate_covariance_matrix(daily_returns)
    monte_carlo_results = monte_carlo_simulation(expected_returns, covariance_matrix, num_portfolios, risk_free_rate)
    optimized_weights_np = optimize_sharpe_ratio(expected_returns, covariance_matrix, risk_free_rate)
    # Convert the NumPy array to a Python list
    optimized_weights_list = optimized_weights_np.tolist()
    # Convert tickers and weights to a dictionary
    optimized_weights = dict(zip(tickers, optimized_weights_list))
    top_portfolios = monte_carlo_results.sort_values(by='sharpe_ratio', ascending=False).head(5)
    weights_dict = dict(zip(tickers, optimized_weights))

    results = {
        "optimized_weights": weights_dict,
        "top_portfolios": top_portfolios,
        "monte_carlo_results": monte_carlo_results,
        "plot_path": plot_path
    }
    
    return results


#lets calculate the downside volatility, we need to replace the cases where returns exceeded the target return with 0, because we only care about when returns were less than target
def calculate_downside_deviation(daily_returns, weights, target_return=0.0):
    portfolio_returns = daily_returns.dot(weights)
    downside_diff = target_return - portfolio_returns
    downside_diff[downside_diff < 0] = 0
    downside_deviation = np.sqrt(np.mean(np.square(downside_diff)))
    
    return downside_deviation

def main():
    portfolio = ['BTC-USD', 'TSLA', 'AMD', 'MSFT', 'AMZN']  
    start_date = '2020-01-01'
    end_date = '2023-01-01'
    num_portfolios = 10000
    risk_free_rate = 0.02
    stock_data = fetch_stock_data(portfolio, start_date, end_date)
    daily_returns = calculate_daily_returns(stock_data)
    expected_returns = calculate_expected_returns(daily_returns)
    covariance_matrix = calculate_covariance_matrix(daily_returns)
    monte_carlo_results = monte_carlo_simulation(expected_returns, covariance_matrix, num_portfolios, risk_free_rate)
    
    weights = np.array([1 / len(portfolio)] * len(portfolio))
    optimized_weights = optimize_sharpe_ratio(expected_returns, covariance_matrix, risk_free_rate)
    
    weights_dict = {ticker: weight for ticker, weight in zip(portfolio, optimized_weights)}

    portfolio_variance = calculate_portfolio_variance(weights, covariance_matrix)
    expected_portfolio_return = np.sum(weights * expected_returns)
    sharpe_ratio = (expected_portfolio_return - risk_free_rate) / np.sqrt(portfolio_variance)
    
    downside_deviation = calculate_downside_deviation(daily_returns, weights)
    sortino_ratio = (expected_portfolio_return - risk_free_rate) / downside_deviation
    top_portfolios = monte_carlo_results.sort_values(by='sharpe_ratio', ascending=False).head(5)
    
    results = {
        'optimized_weights': weights_dict,
        'expected_returns': expected_returns.tolist(),
        'covariance_matrix': covariance_matrix.values.tolist(),
        'portfolio_variance': portfolio_variance,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'top_portfolios': top_portfolios,
        'plot_path': plot_path
    }

    return results

# Call the main function
if __name__ == "__main__":
    main()