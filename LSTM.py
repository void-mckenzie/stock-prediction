# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 15:54:07 2020

@author: mukmc
"""
from nsepy import get_history as gh
import datetime as dt

start = dt.datetime(2020,7,20)
end = dt.datetime(2020,7,22)
stk_data = gh(symbol='NIFTY 50',start=start,end=end)


###...........................................................

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dat = pd.read_csv("data/nifty.csv",index_col=False)

dat['hl']=dat['High']-dat['Low']

#training_set = dat.iloc[0:int((4862)*0.8),4]
training_set = dat.iloc[0:int((4862)*0.8),1:6]

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
#training_set_scaled = sc.fit_transform(np.array(training_set).reshape(-1,1))
training_set_scaled = sc.fit_transform(training_set)

X_train = []
y_train = []
for i in range(30, 3889):
    X_train.append(training_set_scaled[i-30:i, :])
    y_train.append(training_set_scaled[i, 3])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 5))


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


regressor = Sequential()
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 5)))
regressor.add(Dropout(0.1))
regressor.add(LSTM(units = 40,return_sequences = False))
regressor.add(Dense(units = 1))
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(X_train, y_train, epochs = 5, batch_size = 32)


test_set = dat.iloc[int((4862)*0.8):,1:6]

test_set_scaled = sc.transform(test_set)

X_test = []
for i in range(30, 973):
    X_test.append(test_set_scaled[i-30:i, :])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 5))

predicted_stock_price = regressor.predict(X_test)

dummy_test = test_set.copy()
dummy_test['Close'] = np.append(np.arange(0,30),predicted_stock_price)
predicted_stock_price = sc.inverse_transform(dummy_test)

plt.plot(list(test_set['Close'])[30:], color = 'red', label = 'Real NIFTY Stock Price')
plt.plot(predicted_stock_price[30:,3], color = 'blue', label = 'Predicted NIFTY Stock Price')
plt.title('NIFTY Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('NIFTY Stock Price')
plt.legend()
plt.show()

np.mean(np.sum(np.absolute(predicted_stock_price[45:,3] - np.array(test_set['Close'])[45:])))

np.sqrt(((predicted_stock_price[30:,3] - np.array(test_set['Close'])[30:]) ** 2).mean())