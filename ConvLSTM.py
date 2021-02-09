import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.optimizers import Adadelta
from keras.models import model_from_json
from datetime import date, timedelta
from matplotlib import pyplot as plt
import math
from get_data import get_stock_data

import tensorflow as tf
#import keras.backend.tensorflow_backend as tfback

from numpy.random import seed
seed(1)
from tensorflow.compat.v1 import set_random_seed
set_random_seed(2)
#tf.debugging.set_log_device_placement(True)
#tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("tf.__version__ is", tf.__version__)
print("tf.keras.__version__ is:", tf.keras.__version__)

'''
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

'''
def reshape_actual(actual_data):


    shaped_data = [actual_data[i][1] for i in range(len(actual_data))]
    shaped_data = np.concatenate(shaped_data, axis=0)
    shaped_data = shaped_data.reshape((shaped_data.shape[0], 1))
    return shaped_data


def plot_them_graphs(actual, predicted, type, ticker):
    pred = np.zeros((actual.shape[0], 18))
    pred[:, 3] = actual[:, 0]
    pred = np.around(scaler.inverse_transform(pred), decimals=2)

    pred1 = np.zeros((actual.shape[0], 18))
    pred1[:, 3] = predicted[:, 0]
    pred1 = np.around(scaler.inverse_transform(pred1), decimals=2)

    plt.figure()
    plt.plot(pred[:, 3], label='actual')
    plt.plot(pred1[:, 3], label='predicted')
    plt.legend()
    #fig1 = plt.gcf()
    plt.show()
    plt.savefig(ticker + '_' + type + "1.png", transparent=False)


print("tf.__version__ is", tf.__version__)
print("tf.keras.__version__ is:", tf.keras.__version__)

#tfback._get_available_gpus = _get_available_gpus

ticker = input("Please enter symbol: ")
present_date = date.today()
prev_date = date.today() - timedelta(days = 5457)
print(date(present_date.year, present_date.month, present_date.day))
get_stock_data(ticker,start_date = prev_date, end_date = present_date)

dataset = pd.read_csv('stock_prices' + ticker + '.csv')


scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(dataset.values)

##### TRAIN TEST SPLITTING #####
train_gen = TimeseriesGenerator(scaled, scaled[:,3], start_index = 0, end_index = int(len(scaled) * 0.95), length = 1, batch_size = 256)
test_gen = TimeseriesGenerator(scaled, scaled[:,3], start_index = int(len(scaled) * 0.95), end_index = int(len(scaled) - 1), length = 1, batch_size = 256)

##### MODEL CREATION ######
model = Sequential()
model.add(Conv1D(18, kernel_size=3, activation='relu', padding = 'valid', strides=1, input_shape=(1,18), data_format='channels_first'))
model.add(Conv1D(18, kernel_size=3, activation='relu', padding = 'valid', strides=1))
#model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100, activation = 'tanh', recurrent_activation = 'sigmoid', unroll = False, use_bias = True, recurrent_dropout = 0, return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(100, activation = 'tanh', recurrent_activation = 'sigmoid', unroll = False, use_bias = True, recurrent_dropout = 0,return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(100, activation = 'tanh', recurrent_activation = 'sigmoid', unroll = False, use_bias = True, recurrent_dropout = 0,return_sequences= True))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1, activation = 'linear'))
adadelta = Adadelta(learning_rate=1.0, rho=0.95)
model.compile(loss= 'mse', optimizer = adadelta, metrics=['accuracy'])
model.summary()


##### TRAINING #####
history = model.fit_generator(train_gen, epochs = 3000, verbose = 2, shuffle = False, validation_data = test_gen)


##### PLOTTING LOSS ######
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
score = model.evaluate_generator(test_gen, verbose = 1)
print()
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print()

###### RESHAPE ACTUAL DATA #######
actual_train = reshape_actual(train_gen)
predictions_train = model.predict_generator(train_gen, verbose = 0)

##### RSME FOR TRAIN #####
rmse_train = math.sqrt(mean_squared_error(actual_train[:], predictions_train[:]))
print()
print(rmse_train)

######PLOT TRAIN######

plot_them_graphs(actual_train, predictions_train, "train", ticker)

###### TEST DATA ######
actual_test = reshape_actual(test_gen)
predictions_test = model.predict_generator(test_gen, verbose = 0)
rmse_test = math.sqrt(mean_squared_error(actual_test[:], predictions_test[:]))
print()
print(rmse_test)


###### PLOT TEST ######

plot_them_graphs(actual_test, predictions_test, "test", ticker)


##### SAVE IT!!!!!! #####

model_json = model.to_json()
with open("model_" + ticker + "1.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model_" + ticker + "1.h5")
print("Saved model to disk")