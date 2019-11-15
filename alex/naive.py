# naive

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = pd.read_csv('../data/data_pressed.csv')

y = data['Yards']
X = data.copy()
del X['Yards']

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = .3)

reg = LinearRegression().fit(xtrain, ytrain)
ann = MLPRegressor(hidden_layer_sizes = (10,5)).fit(xtrain, ytrain)

reg_hat = reg.predict(xtest)
ann_hat = ann.predict(xtest)

print(mean_squared_error(reg_hat, ytest))
print(mean_squared_error(ann_hat, ytest))