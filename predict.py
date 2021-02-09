import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, LSTM
#from keras.layers import RepeatVector
#from keras.layers import TimeDistributed
#from keras.optimizers import Adadelta
from keras.models import model_from_json
from datetime import date, timedelta
from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import LabelEncoder
import nsepy as nse
import talib
import tensorflow as tf
import keras.backend.tensorflow_backend as tfback

def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]


print("tf.__version__ is", tf.__version__)
print("tf.keras.__version__ is:", tf.keras.__version__)

tfback._get_available_gpus = _get_available_gpus

print()
ticker = input("Enter the Symbol: ")

json_file = open('model_' + ticker + '.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model_" + ticker + ".h5")
print("Loaded model from disk")
model.summary()

today = date.today()
days = timedelta(62)
period = today - days

data = nse.get_history(symbol=ticker, start= period, end= today)
#, 'Volume'
data = data.drop(['Symbol', 'Series', 'Prev Close', 'Last', 'Deliverable Volume', '%Deliverble', 'Trades'], axis = 1)

close = data['Close'].values
high = data['High'].values
low = data['Low'].values

data['upB'], data['midB'], data['lowB'] = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
data['ADX'] = talib.ADX(high, low, close, timeperiod=14)
data['AroonUp'], data['AroonDown'] = talib.AROON(high, low, timeperiod=14)
data['RSI'] = talib.RSI(close, timeperiod=10)
data['K'],d = talib.STOCH(high, low, close, fastk_period=14, slowk_period=14, slowk_matype=0, slowd_period=3, slowd_matype=0)

macd, macdsignal, data['MACD'] = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
data['EMA'] = talib.EMA(close, timeperiod=30)
data['diff'] = data['High'] - data['Low']
data = data.dropna()

req_data = data.values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(req_data)

print()
to_pred = scaled[-1]
to_pred = to_pred.reshape((1, 1, scaled.shape[1]))
print()
######PREDICTION#######

pred  = model.predict(to_pred)

#print(pred)
pred1 = np.zeros((to_pred.shape[0],18))
pred1[:,3] = pred[0]
pred1 = np.around(scaler.inverse_transform(pred1), decimals = 2)
print("The predicted value for the next trading day for " + ticker + " is : " + str(pred1[0,3]))