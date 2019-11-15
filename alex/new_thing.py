import pandas as pd
from sklearn.model_selection import train_test_split

# new idea:
# use a network to turn the collection of player information into a single 
# useful scalar
data = pd.read_csv('../data/data_no_bs.csv')
print(data.columns.values)
print(data.index.values)

mat_list = []
yard_list = []

# for each play:
for play in set(data['PlayId']):
	subset = data[data['PlayId'] == play]

	# capture matrix of player distribution
	mat = subset.loc['X':'Dir']
	yard = subset['Yards'].iloc[0]

	mat_list.append(mat)
	yard_list.append(yard)

# convert matrix to vector
row_list = []
for mat in mat_list:
	row_list.append(np.reshape(mat.values, mat.shape[0] * mat.shape[1]))

# collection of these vectors is data
X = row_list
y = yard_list

# train test split
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = .33)

# train NN model on this