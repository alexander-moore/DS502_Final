# naughty naughty encoding
# actually no just gonna learn keras here

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
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

x_dat, x_test, y_dat, y_test = train_test_split(mms_data, target, test_size = .2)
x_train, x_val, y_train, y_val= train_test_split(x_dat, y_dat, test_size = .25)

model = Sequential()

model.add(Dense(units = 64, activation = 'relu', input_dim = x_train.shape[1]))
model.add(Dense(units = 64, activation = 'relu'))
model.add(Dense(units = 64, activation = 'relu'))
model.add(Dense(units = 1, activation = 'linear'))

atom = Adam(learning_rate = .01)

model.compile(loss = 'MSE', optimizer = atom)

history = model.fit(x_train, y_train,
                epochs=50,
                verbose = 0,
                batch_size=256,
                shuffle=True,
                validation_data=(x_val, y_val))

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

loss_and_metrics = model.evaluate(x_test, y_test, batch_size = 128)

classes = model.predict(x_test, batch_size = 128)

# note
