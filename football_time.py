# foot ball time, baby!

import pandas as pd
import numpy as np
import seaborn as sns
import datetime

raw_data = pd.read_csv('data/train.csv')
print(raw_data.shape)

data = raw_data.drop(columns = ['JerseyNumber', 'Week', 'HomeTeamAbbr', 'VisitorTeamAbbr', 'Week', 'PlayerBirthDate', 'Season', 'Humidity', 'WindSpeed', 'WindDirection'])

# turn 2 useless columns (UTC handoff time, UTC snap time) into a differnce
print(data['TimeHandoff'])

#data['time_to_hand'] = data['TimeHandoff'] - data['TimeSnap']

# homegame?



## data exploration

desc = data.describe()
it = desc.shape[1]

for i in range(it):
	print(desc[desc.columns[i]])

cor_mat = data.corr()

names = data.columns.values

print(names)

cor_mat.style.background_gradient()

print(cor_mat)

sns.pairplot(data)


grumbo = data.isnull().sum(axis = 0)

for ele in grumbo:
	print(ele)

# summarization

# pairplots