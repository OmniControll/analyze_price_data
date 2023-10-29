import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import pandas as pd
import plotly.express as px
import ta

#This code is modular, with functions handling specific tasks.
#The goal is to handle data fetching, financial calculations, optimization, and visualization.
#The main function will ensure the sequence of operations

#first lets get the stock data from yfinance and do some basic financial calculations

def fetch_stock_data(tickers, start_date, end_date):
    stock_data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

    sectors, technical_indicators, market_caps, RSIs, MACDs, moving_averages, moving_averages_200 = {}, {}, {}, {}, {}, {}, {}

    for ticker in tickers:
        ticker_info = yf.Ticker(ticker).info

        sectors[ticker] = ticker_info.get('sector', 'N/A')
        technical_indicators[ticker] = ticker_info.get('fiftyTwoWeekHigh', 'N/A')
        market_caps[ticker] = ticker_info.get('marketCap', 'N/A')
        
        # Get historical data for calculations
        hist_data = yf.download(ticker, start=start_date, end=end_date)
        
        # Calculate RSI
        RSIs[ticker] = ta.momentum.RSIIndicator(close=hist_data['Close']).rsi().iloc[-1] #RSI is a momentum indicator that measures the magnitude of recent price changes to evaluate overbought or oversold conditions in the price of a stock or other asset
        
        # Calculate MACD
        macd = ta.trend.MACD(close=hist_data['Close']) #MACD is a trend-following momentum indicator that shows the relationship between two moving averages of a security’s price
        MACDs[ticker] = macd.macd_diff().iloc[-1] #macd_diff() calculates the difference between the MACD and the signal line
        
        # Calculate moving averages
        moving_averages[ticker] = hist_data['Close'].rolling(window=50).mean().iloc[-1] #rolling() calculates the moving average
        moving_averages_200[ticker] = hist_data['Close'].rolling(window=200).mean().iloc[-1] #rolling() calculates the moving average

    return {
        'stock_data': stock_data,
        'sectors': sectors,
        'technical_indicators': technical_indicators,
        'market_caps': market_caps,
        'RSIs': RSIs,
        'MACDs': MACDs,
        'moving_averages': moving_averages,
        'moving_averages_200': moving_averages_200
    }

def calculate_daily_returns(stock_data): 
    daily_returns = stock_data.pct_change().dropna() #pct_change() calculates the percentage change between the current and prior element
    return daily_returns

def calculate_expected_returns(daily_returns): 
    return daily_returns.mean() # average of the values in the array

def calculate_covariance_matrix(daily_returns): 
    return daily_returns.cov() #cov() calculates the covariance between the columns

def calculate_sortino_ratio(expected_returns, daily_returns, risk_free_rate): #calculates sortino ratio
    downside_deviation = calculate_downside_deviation(daily_returns, weights, target_return=0.0) #calculate downside deviation
    expected_portfolio_return = np.sum(weights * expected_returns) #calculate expected portfolio return
    sortino_ratio = (expected_portfolio_return - risk_free_rate) / downside_deviation #sortino ratio
    return sortino_ratio

def calculate_max_drawdown(daily_returns): #calculates max drawdown
    cumulative_returns = (1 + daily_returns).cumprod() #cumprod() calculates the cumulative product of the array
    max_drawdown = np.max(np.maximum.accumulate(cumulative_returns) - cumulative_returns) #max drawdown
    return max_drawdown

def calculate_value_at_risk(daily_returns, weights, confidence_level=0.05): #calculates value at risk
    portfolio_returns = daily_returns.dot(weights) #take dot product of weights and returns
    value_at_risk = np.quantile(portfolio_returns, confidence_level) #value at risk
    return value_at_risk


def calculate_portfolio_variance(weights, covariance_matrix): #calculates portfolio variance
    return np.dot(weights.T, np.dot(covariance_matrix, weights)) #np.dot() calculates the dot product of two arrays
    
#monte carlo simulation 
def monte_carlo_simulation(expected_returns, covariance_matrix, num_portfolios, risk_free_rate): #simulates random portfolios
    num_assets = len(expected_returns) #number of assets in our portfolio
    results = np.zeros((num_portfolios, num_assets + 3))  # +3 for return, volatility, and Sharpe ratio (3 columns)

    for i in range(num_portfolios): #loop through each portfolio
        # random weights for our portfolio simulation
        weights = np.random.random(num_assets)  #generate random weights
        weights /= np.sum(weights) #normalize weights so they add up to 1

        # Calculations portfolio return, volatility, and Sharpe ratio
        portfolio_return = np.sum(weights * expected_returns) 
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights))) #portfolio variance
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility #sharpe ratio

        # store the results
        results[i, 0:num_assets] = weights  #store the weights
        results[i, num_assets] = portfolio_return  #return is the sum of the weights times the expected returns
        results[i, num_assets + 1] = portfolio_volatility  #volatility is the standard deviation of returns
        results[i, num_assets + 2] = sharpe_ratio #sharpe ratio is the return of the portfolio minus the risk free rate, divided by the volatility

    # Create a Df
    columns = [f'weight_{asset}' for asset in expected_returns.index] + ['return', 'volatility', 'sharpe_ratio'] #column names
    results_df = pd.DataFrame(results, columns=columns) 

    return results_df

def optimize_sharpe_ratio(expected_returns, covariance_matrix, risk_free_rate):
    num_assets = len(expected_returns)
# to calculate the optimized (maximized) sharpe ratio, we need to minimize the negative of it. 
# the function below calculates the var. and return, then outputs the negated sharpe ratio
    def objective_function(weights): #objective function to minimize the negative sharpe ratio
        portfolio_variance = calculate_portfolio_variance(weights, covariance_matrix)  #calculate portfolio variance
        expected_portfolio_return = np.sum(weights * expected_returns)  #calculate expected portfolio return
        neg_sharpe_ratio = - (expected_portfolio_return - risk_free_rate) / np.sqrt(portfolio_variance) #negated sharpe ratio
        return neg_sharpe_ratio
    
# we want constraints: the sum of the weights must equal 100%. we use a lambda with x as argument to represent the weights of our assets in the portfolio
    sum_weights_is_one = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    max_single_weight = {'type': 'ineq', 'fun': lambda x: 0.25 - np.max(x)} #max weight of any single asset is 25% (new constraint)
    constraints = [sum_weights_is_one, max_single_weight] #combine the constraints into a list
    #bounds for each weight (0,1)
    bounds = [(0, 1) for _ in range(num_assets)] #list comprehension
    #starting point
    initial_weights = np.array([1 / num_assets] * num_assets) #initialize weights to 1/num_assets
    optimized_weights = minimize(objective_function, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return optimized_weights.x

def plot_new_metrics(results):
    # Create a DataFrame for the new metrics
    new_metrics_df = pd.DataFrame({
        'Sortino Ratio': [results['sortino_ratio']],
        'Max Drawdown': [results['max_drawdown']],
        'Value at Risk': [results['value_at_risk']]
    })
    
    # Create a bar chart using Plotly
    fig = px.bar(new_metrics_df, 
                 title="Additional Portfolio Metrics", 
                 labels={"value": "Metrics Value", "index": "Metrics"})
    fig.show()

# Now lets run our functions to analyze and optimize our stock portfolio. we'll use a risk free rate of 2%.
# We also list the simulated portfolios by their Sharpe ratio in descending order, and select the top 5..

def analyze_stocks(tickers, start_date, end_date, num_portfolios, risk_free_rate): #analyzes stocks
    fetched_data = fetch_stock_data(tickers, start_date, end_date) #fetch stock data
    stock_data = fetched_data['stock_data'] #get stock data
    daily_returns = calculate_daily_returns(stock_data) #calculate daily returns
    expected_returns = calculate_expected_returns(daily_returns) #calculate expected returns
    covariance_matrix = calculate_covariance_matrix(daily_returns) #calculate covariance matrix

    # New metrics calculations
    optimized_weights_np = optimize_sharpe_ratio(expected_returns, covariance_matrix, risk_free_rate) #optimize sharpe ratio
    sortino_ratio = calculate_sortino_ratio(expected_returns, daily_returns, risk_free_rate) #calculate sortino ratio
    max_drawdown = calculate_max_drawdown(daily_returns) #calculate max drawdown
    value_at_risk = calculate_value_at_risk(daily_returns, optimized_weights_np) #calculate value at risk

    monte_carlo_results = monte_carlo_simulation(expected_returns, covariance_matrix, num_portfolios, risk_free_rate) #monte carlo simulation
    top_portfolios = monte_carlo_results.nlargest(5, 'sharpe_ratio')    #top 5 portfolios based on sharpe ratio

    results = {
        'optimized_weights': dict(zip(tickers, optimized_weights_np.tolist())),
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'value_at_risk': value_at_risk,
        'top_portfolios': top_portfolios,
        'monte_carlo_results': monte_carlo_results
    }
    
    return results

#for downside volatility, we need to replace the cases where returns exceeded the target return with 0, 
# because we only care about when returns were less than target
#downside deviation is a risk measure that focuses on returns that fall below a minimum threshold or target return level
def calculate_downside_deviation(daily_returns, weights, target_return=0.0):
    portfolio_returns = daily_returns.dot(weights)  # take dot product of weights and returns, the dot product is, simply put, a measure of similarity between two vectors
    downside_diff = target_return - portfolio_returns #calculate the difference between the target return and the portfolio return
    downside_diff[downside_diff < 0] = 0  # replace negative values with 0
    downside_deviation = np.sqrt(np.mean(np.square(downside_diff))) #square the difference, take the mean, and take the square root
    return downside_deviation

#visualize results with plotly`s scatter plot`
def plot_monte_carlo_results(monte_carlo_results, optimized_weights, optimized_return, optimized_volatility):
    # Create a Df includes the portfolio weights
    #the lambda function adds a new column to the Df that contains the weights of each asset, formatted to two decimal places
    monte_carlo_results['text'] = monte_carlo_results.apply(lambda row: ', '.join([f"{ticker}: {row[f'weight_{ticker}']:.2f}" for ticker in optimized_weights.keys()]), axis=1)
    monte_carlo_results['return'] *= 100 #convert to percentage
    monte_carlo_results['volatility'] *= 100 #convert to percentage
    #ensure the hover data shows the weights  (text column)
    fig = px.scatter(monte_carlo_results, x="volatility", y="return", color="sharpe_ratio",
                     hover_data={"text": True, 'return': ':.2f%', 'volatility': ':.2f%', 'sharpe_ratio': ':.2f%'}, #formatting
                     labels={'text': "Portfolio Weights"})
    fig.update_layout(xaxis_title="Volatility (%)", yaxis_title="Return (%)") #formatting                 
    # Add marker for optimized portfolios
    fig.add_scatter(x=[optimized_volatility], y=[optimized_return], mode='markers', 
                    marker=dict(size=[30], color=['blue']), #formatting
                    hovertext=[', '.join([f"{ticker}: {weight:.2f}" for ticker, weight in optimized_weights.items()])], #formatting
                    name="Optimized Portfolio")
    print("About to show figure...") #debugging
    fig.show()
    print("Figure should be displayed.") #debugging


#a few things to note about the basket of stocks to pick:
# correlation: different stocks in the same industry tend to move together, so we want to pick stocks that are not highly correlated
# size of basket: we want to pick stocks that are not too correlated, but we also want to pick enough stocks to diversify our portfolio
def main():
    portfolio = ['TSLA', 'COIN', 'GOOGL']
    start_date = '2020-01-01'
    end_date = '2023-10-28'
    num_portfolios = 10000
    risk_free_rate = 0.02

    results = analyze_stocks(portfolio, start_date, end_date, num_portfolios, risk_free_rate)
    print("Top 5 portfolios based on Sharpe Ratio:\n", results['top_portfolios'])
    
    plot_monte_carlo_results(results['monte_carlo_results'], 
                             results['optimized_weights'], 
                             results['top_portfolios'].iloc[0]['return'], 
                             results['top_portfolios'].iloc[0]['volatility'])
                             
    plot_new_metrics(results)

if __name__ == "__main__":
    main()
