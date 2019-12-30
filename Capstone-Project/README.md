# Proposal

## Background
Investment and trading

Knowing the future of a market stock price is the most certain way of obtaining information about risk of investment and is commonly pursued because of potentially infinite profits.

This led me to spend a few years working on trading bots myself. I had learned that trading bot efficiency is short lived and extremely prone to overfitting to historical data which makes them useless in real world predictions applications. But all of my work was developed using hard coded rules or simple linear models. I want to use my experience and improve with a far more advanced model.

Unfortunately market is affected by very large amount of factors that are difficult to implement all together. We can try using NLP on news to gather possible changes on the market or try a technical analysis (which basically comes down to answering "how does the price usually change after these kind of movements?") and many more.

Currency ratio and stock markets forecasting based on technical analysis can be easily implemented in a script thanks to timeseries prediction nature of the problem. Price and volume are publicly accessbile and can be used to generate a set of rules which are aiming to output the most likely future price.

## Problem statement
We want to predict the price of the currency ratio EURUSD and/or stock markets. This is a timeseries forecasting problem. The goal is to create a script that will forecast future values basing on input provided as starting point. Then a web application will serve information live by downloading information from third party services.

## Database and inputs
Our data will consist of price candles entries with columns:
- date -- date of the price candle
- time -- time of the price candle
- open -- price at the candle entry time
- high -- maximum price reached during candle lifespan
- low -- minimum reached during candle lifespan
- close -- candle exit price

This data can be used to calculate indicators like Momentum or Moving Average which can be used as additional features.

Candle lifespan will range from 1 minute to 1 day (if provided only in shorter lifespan a longer will be calculated).

Data is stored in .csv format and to keep it small size I will keep include the history back to year 2015. Timespan of 5 years should be more than enough to train a model.

Dataset will be split into train 60%, validation 20% and test 20%.
Of course data has to be stored chronologically so subsets will be exactly in that order.

### Data sources
- [finance.yahoo.com](finance.yahoo.com) -- live and historical data
- [histdata.com](histdata.com) -- historical data

## Solution statement
To solve this problem I will use PyTorch.
We will build an LSTM model with a 1D Convolution layer as an extension to our features: preprocessed stock value, selected indicators.

## Benchmark model
Moving Average as described [here](https://www.analyticsvidhya.com/blog/2018/10/predicting-stock-price-machine-learningnd-deep-learning-techniques-python/) and Linear Regression will be used as reference models.

## Evaluation metrics
Training set, validation set, test set.
MSE between predicted and actual value.

## Project design
Web application and script for live data predictions.
Feature engineering notebook.
Training notebook and scripts.

%---------------------------------------------------------
# Documentation

## Bibliography:
- [https://towardsdatascience.com/machine-learning-techniques-applied-to-stock-price-prediction-6c1994da8001](https://towardsdatascience.com/machine-learning-techniques-applied-to-stock-price-prediction-6c1994da8001)
- [https://www.analyticsvidhya.com/blog/2018/10/predicting-stock-price-machine-learningnd-deep-learning-techniques-python/](https://www.analyticsvidhya.com/blog/2018/10/predicting-stock-price-machine-learningnd-deep-learning-techniques-python/)

