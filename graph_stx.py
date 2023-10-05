import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.seasonal import seasonal_decompose

register_matplotlib_converters()

# Define the ticker symbol
tickerSymbol = 'BTC-USD'

# Get data on this ticker
tickerData = yf.Ticker(tickerSymbol)

# Get the historical prices for this ticker
tickerDf = tickerData.history(period='1d', start='2020-1-1', end='2023-9-30')

# Plot the close prices
plt.figure(figsize=(10, 8))
plt.plot(tickerDf.Close)
plt.title('Closing price: '+tickerSymbol)
plt.xlabel('Date')
plt.ylabel('Closing Price ($)')
plt.grid(True)
plt.show()

# Moving Average Crossover
short_rolling = tickerDf.Close.rolling(window=20).mean()
long_rolling = tickerDf.Close.rolling(window=100).mean()
plt.figure(figsize=(10,8))
plt.title('Moving Average Crossover: '+tickerSymbol)
plt.plot(short_rolling, label='20 days rolling')
plt.plot(long_rolling, label='100 days rolling')
plt.legend()
plt.grid(True)
plt.show()

# Linear Regression
df = tickerDf.reset_index()
df['timestamp'] = df['Date'].apply(lambda x: x.timestamp())
X = df['timestamp'].values.reshape(-1, 1)
Y = df['Close'].values
model = LinearRegression()
model.fit(X, Y)
trend = model.predict(X)
plt.figure(figsize=(10,8))
plt.title('Linear Regression: '+tickerSymbol)
plt.plot(df['Date'], tickerDf.Close, label='Actual Close Price')
plt.plot(df['Date'], trend, label='Predicted Close Price', color='r')
plt.legend()
plt.grid(True)
plt.show()


print(df.describe())

# Drop Dividends and Stock Splits
df = df.drop(['Dividends', 'Stock Splits'], axis=1)

print(df.info())

# Examine the BTC price's seasonality, trends, and residuals using time-series decomposition
result = seasonal_decompose(df['Close'], model='multiplicative', period=30)
fig = plt.figure()
fig = result.plot()
fig.set_size_inches(10, 8)
plt.show()

# Examine the BTC price's seasonality, trends, and residuals using time-series decomposition with an additive model instead of multiplicative
#to compare the results
result = seasonal_decompose(df['Close'], model='additive', period=30)
fig = plt.figure()
fig = result.plot()
fig.set_size_inches(10, 8)
plt.show()


# Check if the data has a daily or weekly pattern.

# Daily
df['day'] = df['Date'].dt.dayofweek
df.groupby('day')['Close'].mean().plot(kind='bar')
plt.title('Daily Pattern')
plt.xlabel('Day of Week')
plt.ylabel('Average Close Price')
plt.grid(True)
plt.show()
