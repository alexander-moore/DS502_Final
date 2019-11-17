# foot ball time, baby!

import pandas as pd
import numpy as np
import seaborn as sns
import datetime


raw_data = pd.read_csv('../data/train.csv')
print(raw_data.shape)

data = raw_data.drop(columns = ['JerseyNumber', 'Week', 'HomeTeamAbbr', 'VisitorTeamAbbr', 'Week', 'PlayerBirthDate', 'Humidity', 'WindSpeed', 'WindDirection'])
data = data.drop(columns = ['Team', 'Season', 'DisplayName', 'Location', 'Stadium', 'StadiumType', 'PlayerCollegeName'])

# turn 2 useless columns (UTC handoff time, UTC snap time) into a differnce
print(data['TimeHandoff'])

print(data.columns.values)

#data['time_to_hand'] = data['TimeHandoff'] - data['TimeSnap']

# homegame?



## data exploration

desc = data.describe()
#it = desc.shape[1]

#for i in range(it):
#	print(desc[desc.columns[i]])

#cor_mat = data.corr()

names = data.columns.values

#print(names)

#cor_mat.style.background_gradient()

#print(cor_mat)

#sns.pairplot(data)


grumbo = data.isnull().sum(axis = 0)

for ele in grumbo:
	print(ele)

# summarization

# pairplots

pressed = data.groupby(by = 'PlayId').mean()

#pressed.to_csv('../data/pressed.csv')
data.to_csv('../data/data_no_bs.csv')

#sys.exit()

print(pressed)
print(pressed.shape)
print(pressed.columns.values) # this just removes 

# process plays into observations

# vectors to append onto Pressed:
time_hts = [] #TimeHandoff - TimeSnap
defense_pers = [] #DefensePersonnel
offense_pers = [] #OffensePersonnel
nflid_rusher = [] #NflIdRusher
offense_form = [] #OffenseFormation
game_clock = [] #GameClock
weather = [] #GameWeather

pressed['PlayId'] = pressed.index.values


print(pressed.index)
print(pressed['PlayId'])

for play in list(pressed['PlayId']):
	print(play)
	#print('hi')

	subset = data[data['PlayId'] == play]

	#print(subset)
	#print(subset.columns.values)

	#time_hts.append(subset['TimeHandoff'][0] - subset['TimeSnap'][0])
	defense_pers.append(subset['DefensePersonnel'].iloc[0])
	offense_pers.append(subset['OffensePersonnel'].iloc[0])
	nflid_rusher.append(subset['NflIdRusher'].iloc[0])
	offense_form.append(subset['OffenseFormation'].iloc[0])
	game_clock.append(subset['GameClock'].iloc[0])
	weather.append(subset['GameWeather'].iloc[0])

#pressed.join(time_hts, how = 'right')
pressed.join(defense_pers, how = 'right')
pressed.join(offense_pers, how = 'right')
pressed.join(nflid_rusher, how = 'right')
pressed.join(offense_form, how = 'right')
pressed.join(game_clock, how = 'right')

print(pressed.columns.values)

pressed.to_csv('../data/pressed_appended.csv')