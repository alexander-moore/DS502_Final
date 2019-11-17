
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# new idea:
# use a network to turn the collection of player information into a single 
# useful scalar
data = pd.read_csv('../data/data_no_bs.csv')
print(data.columns.values)
print(data.index.values)

# MIGHT NEED TO INCORPORATE SOMETHING ABOUT PLAYDIRECTION AND OFFENSE VS. DEFENSE (WHICH TEAM EACH PLAYER IS ON)
play_direction = []

mat_list = []
yard_list = []

# for each play:
for play in set(data['PlayId']):
	subset = data[data['PlayId'] == play]
	#print(subset.shape)

	# capture matrix of player distribution
	mat = subset[['X', 'Y', 'S', 'A', 'Dis', 'Orientation', 'Dir', 'PlayerWeight']]
	#print(mat.shape)
	yard = subset['Yards'].iloc[0]
	#print(yard)

	mat_list.append(mat)
	yard_list.append(yard)

print(mat_list[0])
print(len(mat_list))
print(mat_list[0].shape)

# convert matrix to vector
row_list = []
for mat in mat_list:
	row_list.append(np.reshape(mat.values, mat.shape[0] * mat.shape[1]))

row_list = pd.DataFrame(row_list)
row_list.to_csv('row_phys_data.csv', index = False)
yard_list = pd.DataFrame(yard_list)
yard_list.to_csv('yard_data.csv', index = False)
# collection of these vectors is data
X = row_list
print(len(X))
print(X[0].shape)
y = yard_list
print(len(y))

print(type(X))
print(type(X[0]))
print(type(X[0][0]))

print(type(y))
print(type(y[0]))
print(type(y[0][0]))

raw_data['Temperature'] = raw_data['Temperature'].fillna(raw_data['Temperature'].mean())
raw_data['Humidity'] = raw_data['Humidity'].fillna(raw_data['Humidity'].mean())

print('hi')
print(X.isnull().any())
print(y.isnull().any())
print('hi')

sys.exit()
# Now we will use the player distribution data we made to train a network which predicts the effectiveness of the relative distrubution
# this effectiveness will be a new feature to replace the ones used here
data = X
target = y[0]
mms_data = MinMaxScaler().fit_transform(data)

# train test split
x_dat, x_test, y_dat, y_test = train_test_split(mms_data, target, test_size = .2)
x_train, x_val, y_train, y_val= train_test_split(x_dat, y_dat, test_size = .25)

print(x_train.shape)
print(x_val.shape)
print(y_train.shape)
print(y_val.shape)

sys.exit()

model = Sequential()

model.add(Dense(units = 32, activation = 'relu', input_dim = x_train.shape[1]))
model.add(Dense(units = 8, activation = 'relu'))
model.add(Dense(units = 1, activation = 'linear'))

atom = Adam(lr = .005)

model.compile(loss = 'MSE', optimizer = atom)

history = model.fit(x_train, y_train,
                epochs = 500,
                verbose = 1,
                batch_size = 32,
                shuffle = True,
                validation_data = (x_val, y_val))

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

loss_and_metrics = model.evaluate(x_test, y_test, batch_size = 128)

test_player_distribution_score = model.predict(x_test, batch_size = 128)