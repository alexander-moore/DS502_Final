import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.set_printoptions(threshold=np.inf)

raw_data = pd.read_csv('data/train.csv')

print(np.unique(raw_data['PlayId'].values).size)
print(raw_data.columns)

IMAGE_SCALE = 1
px = int(round(60*IMAGE_SCALE))
py = int(round(130*IMAGE_SCALE))

data = pd.DataFrame()
data['id'] = raw_data['PlayId']
data['dir'] = raw_data['PlayDirection']
data['yards'] = raw_data['Yards']
data['x'] = (raw_data['X']*IMAGE_SCALE).round()
data['y'] = (raw_data['Y']*IMAGE_SCALE).round()
data['ox'] = np.sin(np.deg2rad(raw_data['Orientation']))
data['oy'] = np.cos(np.deg2rad(raw_data['Orientation']))
data['dx'] = raw_data['Dis'] * np.sin(np.deg2rad(raw_data['Dir']))
data['dy'] = raw_data['Dis'] * np.cos(np.deg2rad(raw_data['Dir']))
data['vx'] = raw_data['S'] * np.sin(np.deg2rad(raw_data['Dir']))
data['vy'] = raw_data['S'] * np.cos(np.deg2rad(raw_data['Dir']))
data['ax'] = raw_data['A'] * np.sin(np.deg2rad(raw_data['Dir']))
data['ay'] = raw_data['A'] * np.cos(np.deg2rad(raw_data['Dir']))
data['offense'] = np.equal(raw_data['Team'].values, 'home') * np.equal(raw_data['HomeTeamAbbr'].values, raw_data['PossessionTeam'].values) + np.equal(raw_data['Team'].values, 'away') * np.equal(raw_data['VisitorTeamAbbr'].values, raw_data['PossessionTeam'].values)
data['defence'] = data['offense'].apply(lambda x: not x)
data['direction'] = np.equal(raw_data['PlayDirection'],'right')
data['height'] = raw_data['PlayerHeight'].apply(lambda x: [int(n)*12 for n in x.split('-')][0] + [int(n) for n in x.split('-')][1])
data['weight'] = raw_data['PlayerWeight']
data['age'] = raw_data['PlayerBirthDate'].apply(lambda x: 2019 - int(x.split('/')[2]))

ss = StandardScaler()
norm_data = data[['dx','dy','vx','vy','ax','ay','height','weight','age']]
data[norm_data.columns] = ss.fit_transform(norm_data)

field = []
yards = []
count = 0
overlap = 0
for _,play in data.groupby(['id']):
    count += 1
    if play[['x','y']].pivot_table(index=['x','y'],aggfunc='size').max() > 1:
        overlap += 1
    try:
        f = np.zeros((px+1,py+1,14))
    except MemoryError:
        print(px)
        print(py)
        print(count)
        exit()
    for i,d in enumerate(play.drop(['id','dir','yards','x','y'],axis=1).columns):
        for r,c,v in zip(play['y'].values, play['x'].values, play[d].values):
            if play['dir'].values[0] == 'right':
                f[int(r),int(c),i] += v
            else:
                f[int(r),py-int(c),i] += v
    field.append(f)
    yards.append(play['yards'].values[0])
print(str(overlap) + ' out of ' + str(np.unique(data['id'].values).size) + ' plays with overlap.')

field = np.stack(field)
yards = np.array(yards)
field_trn, field_val, yards_trn, yards_val = train_test_split(field, yards, train_size=0.75)
print(field_trn.shape)
print(yards_trn.shape)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1, 1),activation='relu',input_shape=(field_trn.shape[1:])))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1, 1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1, 1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=64,activation='relu'))
model.add(Dense(units=1,activation='linear'))
model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
callbacks = [EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=False)]

model.fit(x=field_trn,y=yards_trn,epochs=1000,verbose=1,callbacks=callbacks,validation_data=(field_val,yards_val))

print(np.stack(model.predict(field_val).tolist(),yards_val.tolist()))