# -*- coding: utf-8 -*-
"""
Created on Thu Feb  10 14:15:59 2022

@author: kiero
"""
import math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import pandas as pd
import pandas_datareader as pdr

import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

#Load Data
company = '^GSPC'

start = dt.datetime(2012, 1, 1)
end = dt.datetime(2017, 1, 1)

df = pdr.DataReader(company, 'yahoo', start, end)
print(df.tail(5))

plt.figure(figsize=(15,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Value', fontsize=18)
plt.show()

data = df.filter(['Close'])
dataset = data.values
training_data_len = math.ceil(len(dataset)*.8)

#print(training_data_len)
#prepare data and scale
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:training_data_len, :]
x_train=[]
y_train=[]

prediction_days = 60

for x in range(prediction_days, len(train_data)):
    x_train.append(train_data[x-prediction_days:x, 0])
    y_train.append(train_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


#build the LSTM
model = Sequential()
    
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dropout(0.2))
model.add(Dense(1))


model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=16)

test_data = scaled_data[training_data_len - prediction_days: , :]
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(prediction_days, len(test_data)):
    x_test.append(test_data[i-prediction_days:i,0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
#print(rmse)

train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

plt.figure(figsize=(15,8))
plt.title('LSTM Model')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Value', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid['Predictions'])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

#Predict Next Day

SP_quote = pdr.DataReader('^GSPC', data_source='yahoo', start = '2012-01-01', end= '2016-12-31')
new_df = SP_quote.filter(['Close'])
last_60_days  = new_df[-60:].values
scaler.fit(last_60_days)
last_60_days_scaled = scaler.transform(last_60_days)
X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)