import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
import seaborn as sns


register_matplotlib_converters()

# Define the ticker symbol
tickerSymbol = 'BTC-USD'

# Get data on this ticker
tickerData = yf.Ticker(tickerSymbol)

# Get the historical prices for this ticker
tickerDf = tickerData.history(period='1d', start='2016-1-1', end='2023-9-30')

# Drop Dividends and Stock Splits because we're dealing with a cryptocurrency
tickerDf = tickerDf.drop(['Dividends', 'Stock Splits'], axis=1)

#transform target variable (log closing price), to logaritmic scale because we're doign time series analysis
tickerDf['Log_Close'] = np.log(tickerDf['Close'])

print(tickerDf.describe())

#EDA

# Plot the close prices
plt.figure(figsize=(10, 8))
sns.lineplot(x = 'Date', y= 'Close', data = tickerDf)
plt.title('Log of Closing Price: '+tickerSymbol)
plt.xlabel('Date')
plt.ylabel('Log of Closing Price')
plt.grid(True)
plt.show()


# Moving Average Crossover
short_rolling = tickerDf['Log_Close'].rolling(window=20).mean()
long_rolling = tickerDf['Log_Close'].rolling(window=100).mean()
plt.figure(figsize=(10,8))
sns.lineplot(data = short_rolling, label='20-day rolling mean')
sns.lineplot(data = long_rolling, label='100-day rolling mean')
plt.legend()
plt.grid(True)
plt.show()

# Exponential Moving Average Crossover
ema_short = tickerDf['Log_Close'].ewm(span=20, adjust=False).mean()
ema_long = tickerDf['Log_Close'].ewm(span=100, adjust=False).mean()
plt.figure(figsize=(10,8))
sns.lineplot(data = ema_short, label='20-day EMA')
sns.lineplot(data = ema_long, label='100-day EMA')
plt.legend()
plt.grid(True)
plt.show()


# Linear Regression
tickerDf = tickerDf.reset_index()
tickerDf['timestamp'] = tickerDf['Date'].apply(lambda x: x.timestamp())
X = tickerDf['timestamp'].values.reshape(-1, 1)
Y = tickerDf['Log_Close'].values
log_model = LinearRegression()
# train the beast
log_model.fit(X, Y)
log_trend = log_model.predict(X)

# Plot the beast 
plt.figure(figsize=(10,8))
plt.title('Linear Regression: '+tickerSymbol)
sns.lineplot(x=tickerDf['Date'], y=tickerDf['Log_Close'], label='Log Closing Price')
sns.lineplot(x=tickerDf['Date'], y=log_trend, label='Predicted Log Closing Price')
plt.xlabel('Date')
plt.ylabel('Log Closing Price')
plt.legend()
plt.grid(True)
plt.show()

# the line looks to be quite underfit. This is because the linear regression model is not able to capture the seasonality of the data.

#  time-series decomposition: to decompose the time series into its components: seasonality, trends, and residuals

result = seasonal_decompose(tickerDf['Log_Close'], model='multiplicative', period=30)
fig = plt.figure()
fig = result.plot()
fig.set_size_inches(10, 8)
plt.show()
#Residual is the difference between the observed value and the predicted value
#The residual is the part of the data that the model is unable to explain
# th residual is near 1, this means the decomposition may have left out some structure in the data, which makes sense because our line does not fit the data perfectly


# Examine the BTC price's seasonality, trends, and residuals using time-series decomposition with an additive model instead of multiplicative
#to compare the results
result = seasonal_decompose(tickerDf['Close'], model='additive', period=30)
fig = plt.figure()
fig = result.plot()
fig.set_size_inches(10, 8)
plt.show()

# next time i will try to use a different feature engineering technique to see if we can get a better fit for the data
# like polynomial regression, or maybe a different model altogether
