# naughty naughty encoding
# actually no just gonna learn keras here

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('data/clean_census_income.csv')
target = data['age']
del data['age']

scaler = MinMaxScaler()
mms_data = scaler.fit_transform(data)

x_train, x_test, y_train, y_test = train_test_split(mms_data, target, test_size = .3)

model = Sequential()

model.add(Dense(units = 64, activation = 'relu', input_dim = x_train.shape[1]))
model.add(Dense(units = 1, activation = 'linear'))

model.compile(loss = 'MSE', optimizer = 'adam')

model.fit(x_train, y_train, epochs = 5, batch_size = 32)

loss_and_metrics = model.evaluate(x_test, y_test, batch_size = 128)

classes = model.predict(x_test, batch_size = 128)

# note
model.fit(x_train, y_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_val, y_val))