import pandas as pd
import numpy as np
from datetime import date
import nsepy as nse
import talib

def get_stock_data(ticker, start_date, end_date):

    print(end_date, type(ticker))
    data = nse.get_history(symbol=ticker, start=start_date,
                           end= end_date)
    #data = nse.get_history(symbol= ticker, start=start_date, end= end_date)

    data = data.drop(['Prev Close','Symbol', 'Series', 'Deliverable Volume', '%Deliverble', 'Trades', 'Last'], axis = 1)

    close = data['Close'].values
    high = data['High'].values
    low = data['Low'].values

    #Computing technical indicators
    data['upB'], data['midB'], data['lowB'] = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    data['RSI'] = talib.RSI(close, timeperiod=10)
    data['K'], d = talib.STOCH(high, low, close, fastk_period=14, slowk_period=14, slowk_matype=0, slowd_period=3,
                               slowd_matype=0)
    macd, macdsignal, data['MACD'] = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

    data['EMA'] = talib.EMA(close, timeperiod=30)
    data['ADX'] = talib.ADX(high, low, close, timeperiod=14)
    data['AroonUp'], data['AroonDown'] = talib.AROON(high, low, timeperiod=14)
    data['diff'] = data['High'] - data['Low']
    data = data.dropna()


    #saving as a csv file
    data.to_csv('stock_prices' + str(ticker) + '.csv', index=False)