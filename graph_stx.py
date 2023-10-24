import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np

register_matplotlib_converters()  # to avoid warning message

# Define the ticker symbol
tickerSymbol = 'BTC-USD'

# Get data on this ticker
tickerData = yf.Ticker(tickerSymbol)

# Get the historical prices for this ticker
tickerDf = tickerData.history(period='1d', start='2020-1-1', end='2023-10-24')

df = tickerDf.reset_index() # reset the index
df = pd.DataFrame(df) # convert the data to a dataframe
df['timestamp'] = df['Date'].apply(lambda x: x.timestamp()) # convert the date to a timestamp, because the LinearRegression model does not accept datetime data

print(df.head())
print(df.describe())

# Drop Dividends and Stock Splits
df = df.drop(['Dividends', 'Stock Splits'], axis=1)

print(df.info())

# Examine the BTC price's seasonality, trends, and residuals using time-series decomposition
result = seasonal_decompose(df['Close'], model='multiplicative', period=490)
fig = plt.figure()
fig = result.plot()
fig.set_size_inches(10, 8)
plt.show()

# it looks like the BTC price has a strong seasonality, and the trend is not linear.
# so a linear regression model will not work well with this data.

# Examine the BTC price's seasonality, trends, and residuals using time-series decomposition with an additive model instead of multiplicative
#to compare the results
result = seasonal_decompose(df['Close'], model='additive', period=490)
fig = plt.figure()
fig = result.plot()
fig.set_size_inches(10, 8)
plt.show()


# Plot the close prices 
plt.figure(figsize=(10, 8))
plt.plot(tickerDf.Close)
plt.title('Closing price: '+tickerSymbol)
plt.xlabel('Date')
plt.ylabel('Closing Price ($)')
plt.grid(True)
plt.show()

# Moving Average Crossover, with a short window of 20 days and a long window of 100 days
# including the closing price
short_rolling = tickerDf.Close.rolling(window=20).mean() 
long_rolling = tickerDf.Close.rolling(window=100).mean()
plt.figure(figsize=(10,8))
plt.title('Moving Average Crossover: '+tickerSymbol) 
plt.plot(short_rolling, label='20 days rolling', color='y') # yellow
plt.plot(long_rolling, label='100 days rolling', color='r') # red
plt.plot(tickerDf.Close, label='Actual Close Price')
plt.legend()
plt.grid(True)
plt.show()

# split the data into training and test sets
X = tickerDf.drop(['Close'], axis=1)
y = tickerDf['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 80% training and 20% test because we have a lot of data
print('Training set:', X_train.shape, y_train.shape) 
print('Test set:', X_test.shape, y_test.shape)

# Linear Regression

X = df['timestamp'].values.reshape(-1, 1) # reshape the data, because we have only one feature
Y = df['Close'].values # the target variable
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

# evaluating the model
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f' % mean_squared_error(Y, trend))
print('R-score: %.2f' % model.score(X, Y))

# we got a very low R-score, which means that the model is not capturing all of the information in the data.
# for a linear regression model to work well, the data should be linear. In this case, the data is not linear, so we need to use a different model.

# Polynomial Regression
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
poly.fit(X_poly, Y)
model = LinearRegression()
model.fit(X_poly, Y)
trend = model.predict(X_poly)
plt.figure(figsize=(10,8))
plt.title('Polynomial Regression: '+tickerSymbol)
plt.plot(df['Date'], tickerDf.Close, label='Actual Close Price')
plt.plot(df['Date'], trend, label='Predicted Close Price', color='r')
plt.legend()
plt.grid(True)
plt.show()

# evaluating the model
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f' % mean_squared_error(Y, trend))
print('R-score: %.2f' % model.score(X_poly, Y))

# we got a better R-score, but it is still not good enough. We need to use a different model.

