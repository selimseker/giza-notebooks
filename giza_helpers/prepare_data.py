import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_squared_error as mse


def download_data():
    uni_ticker = "UNI-USD"
    eth_ticker = "ETH-USD"
    start = datetime.datetime(2019, 1, 1)
    end = datetime.datetime(2024, 4, 1)
    uni = yf.download(uni_ticker, start=start, end=end, interval="1d")
    eth = yf.download(eth_ticker, start=start, end=end, interval="1d")
    uni = uni.reset_index()
    uni.to_csv("uni.csv", index=False)
    eth = eth.reset_index()
    eth.to_csv("eth.csv", index=False)
    return uni, eth


def process_data(uni: pd.DataFrame, eth: pd.DataFrame):
    uni = uni[uni["Open"] < 0.30]
    uni = uni[["Date", "Open"]]
    eth = eth[["Date", "Open"]]

    uni.rename(columns={"Open": "UNI"}, inplace=True)
    eth.rename(columns={"Open": "ETH"}, inplace=True)

    df = pd.merge(uni, eth, on="Date")
    df.dropna(inplace=True)
    df["price"] = df["ETH"] / df["UNI"]
    ret = 100 * (df["price"].pct_change()[1:])
    print(ret)
    realized_vol = ret.rolling(5).std()
    realized_vol = pd.DataFrame(realized_vol)
    realized_vol.reset_index(drop=True, inplace=True)
    returns_svm = ret**2  # returns squared
    returns_svm = returns_svm.reset_index()
    X = pd.concat([realized_vol, returns_svm], axis=1, ignore_index=True)
    X = X[4:].copy()
    X = X.reset_index()
    X.drop("index", axis=1, inplace=True)
    X.drop(1, axis=1, inplace=True)
    X.rename(columns={0: "realized_vol", 2: "returns_squared"}, inplace=True)
    X["target"] = X["realized_vol"].shift(-5)
    X.dropna(inplace=True)
    Y = X["target"]
    X.drop("target", axis=1, inplace=True)
    n = 252
    X_train = X.iloc[:-n]
    X_test = X.iloc[-n:]
    Y_train = Y.iloc[:-n]
    Y_test = Y.iloc[-n:]
    return X_train, X_test, Y_train, Y_test

