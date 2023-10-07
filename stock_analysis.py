import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import pandas as pd
import plotly.express as px

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
    
#monte carlo simulation 
def monte_carlo_simulation(expected_returns, covariance_matrix, num_portfolios, risk_free_rate):
    num_assets = len(expected_returns)
    results = np.zeros((num_portfolios, num_assets + 3))  # +3 for return, volatility, and Sharpe ratio

    for i in range(num_portfolios):
        # random weights for our portfolio simulation
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)

        # Calculations portfolio return, volatility, and Sharpe ratio
        portfolio_return = np.sum(weights * expected_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

        # store the results
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
# to calculate the optimized (maximized) sharpe ratio, we need to minimize the negative of it. 
# the function below calculates the var. and return, then outputs the negated sharpe ratio
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
    
    optimized_weights_list = optimized_weights_np.tolist()
    
    optimized_weights_dict = dict(zip(tickers, optimized_weights_list))

    top_portfolios = monte_carlo_results.nlargest(10, 'sharpe_ratio')

    results = {
        "optimized_weights": optimized_weights_dict,
        "optimized_return": top_portfolios.iloc[0]['return'],
        "optimized_volatility": top_portfolios.iloc[0]['volatility'],
        "top_portfolios": top_portfolios,  # Now defined
        "monte_carlo_results": monte_carlo_results,
    }
    return results

#for downside volatility, we need to replace the cases where returns exceeded the target return with 0, 
# because we only care about when returns were less than target
def calculate_downside_deviation(daily_returns, weights, target_return=0.0):
    portfolio_returns = daily_returns.dot(weights)
    downside_diff = target_return - portfolio_returns
    downside_diff[downside_diff < 0] = 0
    downside_deviation = np.sqrt(np.mean(np.square(downside_diff)))
    return downside_deviation

#visualize results with new function
def plot_efficient_frontier(monte_carlo_results, optimized_weights, optimized_return, optimized_volatility):
    # Create a Df includes the portfolio weights
    #the lambda function adds a new column to the Df that contains the weights of each asset, formatted to two decimal places
    monte_carlo_results['text'] = monte_carlo_results.apply(lambda row: ', '.join([f"{ticker}: {row[f'weight_{ticker}']:.2f}" for ticker in optimized_weights.keys()]), axis=1)
    monte_carlo_results['return'] *= 100
    monte_carlo_results['volatility'] *= 100 #convert to percentage
    #ensure the hover data shows the weights  (text column)
    fig = px.scatter(monte_carlo_results, x="volatility", y="return", color="sharpe_ratio",
                     hover_data={"text": True, 'return': ':.2f%', 'volatility': ':.2f%', 'sharpe_ratio': ':.2f%'}, #formatting
                     labels={'text': "Portfolio Weights"})
    fig.update_layout(xaxis_title="Volatility (%)", yaxis_title="Return (%)") #formatting                 
    # Add marker for optimized portfolios
    fig.add_scatter(x=[optimized_volatility], y=[optimized_return], mode='markers',
                    marker=dict(size=[30], color=['blue']),
                    hovertext=[', '.join([f"{ticker}: {weight:.2f}" for ticker, weight in optimized_weights.items()])],
                    name="Optimized Portfolio")

    fig.show()


#a few things to note about the basket of stocks to pick:
# correlation: different stocks in the same industry tend to move together, so we want to pick stocks that are not highly correlated
# volatility: we want to pick stocks that are not too volatile, because we want to minimize risk
# size of basket: we want to pick stocks that are not too correlated, but we also want to pick enough stocks to diversify our portfolio

def main():
    portfolio = ['ETH-USD', 'COIN', 'AAPL', 'NEE', 'NVDA', 'JPM', 'DIS', 'AMZN', 'AMD', 'SQ', 'TSLA', 'JNJ'] #add stocks to this list
    start_date = '2020-01-01'
    end_date = '2023-01-01'
    num_portfolios = 10000
    risk_free_rate = 0.02
    results = analyze_stocks(portfolio, start_date, end_date, num_portfolios, risk_free_rate)
    print("Top 5 portfolios based on Sharpe Ratio:\n", results['top_portfolios'])

    return results

if __name__ == "__main__":
    results = main()
    monte_carlo_results = results['monte_carlo_results']
    plot_efficient_frontier(
        monte_carlo_results, 
        results['optimized_weights'], 
        results['optimized_return'], 
        results['optimized_volatility']
    )