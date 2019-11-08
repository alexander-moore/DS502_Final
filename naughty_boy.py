import tensorflow as tf

from tf.keras.models import Sequential
from tf.keras.layers import Dense

import numpy as np

x_input = np.array([[1,2,3,4,5]])
y_input = np.array([[10]])


model = Sequential()
model.add(Dense(units=32, activation="tanh", input_dim=x_input.shape[1], kernel_initializer='random_normal'))
model.add(Dense(units=1, kernel_initializer='random_normal'))

model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

model.summary()
