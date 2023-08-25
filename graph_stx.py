import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Define the ticker symbol
tickerSymbol = 'MSFT'

# Get data on this ticker
tickerData = yf.Ticker(tickerSymbol)

# Get the historical prices for this ticker
tickerDf = tickerData.history(period='1d', start='2010-1-1', end='2023-1-25')

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

# ARIMA
model = ARIMA(tickerDf.Close, order=(5,1,0))
model_fit = model.fit(disp=0)
forecast = model_fit.forecast(steps=30)[0]
plt.figure(figsize=(10,8))
plt.title('ARIMA: '+tickerSymbol)
plt.plot(tickerDf.Close)
plt.plot(pd.date_range(start=tickerDf.index[-1], periods=30, closed='right'), forecast)
plt.grid(True)
plt.show()
