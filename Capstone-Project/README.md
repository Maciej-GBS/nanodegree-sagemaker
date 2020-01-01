# Proposal

## Background
Investment and trading

Knowing the future of a market stock price is the most certain way of obtaining information about risk of investment and is commonly pursued because of potentially infinite profits.

Unfortunately market is affected by very large amount of factors that are difficult to implement all together. We can try using NLP on news to gather possible changes on the market or try a technical analysis and many more. Learning how to efficiently forecast a price requires finding patterns, especially anomalies that indicate a probable future value (James Simons, Numberphile 2015). ML is a great tool for finding patterns in big data and we can use it to solve this forecasting problem.

Currency ratio and stock markets forecasting based on technical analysis can be easily implemented in a script thanks to timeseries prediction nature of the problem. Price and volume are publicly accessbile and can be used to generate a set of rules which are aiming to output the most likely future price.

My personal experience was to work on trading bots myself for a few months. I had learned that trading bot efficiency is short lived and extremely prone to overfitting to historical data which makes them useless in real world predictions applications. But all of my work was developed using hard coded rules or simple linear models. I want to use gained knowlegde to improve with a far more advanced model.

## Problem statement
We want to predict the price of the EURUSD ratio and NASDAQ stock. This is a timeseries forecasting problem. The goal is to create a script that will forecast future values basing on input provided as starting point. Then a web application will serve information live by downloading information from third party services.

## Database and inputs
Our data will consist of price candles entries with columns:
- datetime -- date and time of the price candle opening (may be split into separate columns)
- open -- price at the candle entry time
- high -- maximum price reached during candle lifespan
- low -- minimum reached during candle lifespan
- close -- candle exit price

This data can be used to calculate indicators for example: Momentum, MA, OsMA, MACD. Indicators can be used as additional features.

Candle lifespan will range from 1 minute to 1 day (if provided only in shorter lifespan a longer will be calculated).

Data is stored in .csv format and to keep it small size I will keep include the history back to year 2015. Timespan of 5 years should be more than enough to train a model.

Dataset will be split into train 60%, validation 20% and test 20%.
Of course data has to be stored chronologically so subsets will be exactly in that order.

### Data sources
- [finance.yahoo.com](finance.yahoo.com) -- live and historical data
- [histdata.com](histdata.com) -- historical data

## Solution statement
To solve this problem I will use PyTorch.
We will build an LSTM model and an fully connected model with a 1D Convolution layer as an extension to our features: preprocessed stock value, selected indicators.

## Benchmark model
Moving Average as described [here](https://www.analyticsvidhya.com/blog/2018/10/predicting-stock-price-machine-learningnd-deep-learning-techniques-python/) will be used as reference model.

## Evaluation metrics
Training set, validation set, test set.
MSE between predicted and actual values.

## Project design
Crucial part of the design is to normalize the data.
Stock values can range from 0 to infinity, but we cannot tell our model explicitly about it and it will learn that upper limit is the maximum value from the dataset.
One way to solve this problem is to calculate MA indicator with a length parameter (count of values to calculate mean from) and subtract MA from each price value. MA length becomes a hyperparameter of the model.

Input can consist of a number of recent price values. Controlled by a hyperparameter with minimum 3 candles.

Best model architecture has to be expermientally discovered. I will start with LSTM with 1 unit to predict just next value of the close price.

%---------------------------------------------------------
# Documentation

## Bibliography:
- [https://towardsdatascience.com/machine-learning-techniques-applied-to-stock-price-prediction-6c1994da8001](https://towardsdatascience.com/machine-learning-techniques-applied-to-stock-price-prediction-6c1994da8001)
- [https://www.analyticsvidhya.com/blog/2018/10/predicting-stock-price-machine-learningnd-deep-learning-techniques-python/](https://www.analyticsvidhya.com/blog/2018/10/predicting-stock-price-machine-learningnd-deep-learning-techniques-python/)

