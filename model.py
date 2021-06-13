# Importing the libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
data = pd.read_csv('data.csv')
crop=data.values
X3= crop[:,0:-1]  # slice all rows and start with column 0 and go up to but not including the last column
y3 = crop[:,-1]  # slice all rows and last column, essentially separating out 'crop' column
Y3=y3
le = preprocessing.LabelEncoder() # converting crop column to numeric values
if Y3.dtype == object:
    Y3 = le.fit_transform(Y3)
else:
    pass
train_feature2, test_feature2, train_target2, test_target2 = train_test_split(X3, Y3, test_size=0.25, random_state=1)
print('Observations for Target: %d' % (len(Y3)))
print('Training Observations for Target: %d' % (len(train_target2)))
print('Testing Observations for Target: %d' % (len(test_target2)))
# Create the random forest model and fit to the training data
rfr = RandomForestRegressor(n_estimators=1000)
rfr.fit(train_feature2, train_target2)
print(rfr.score(train_feature2, train_target2))# Look at the R^2 scores on train and test
print(rfr.score(test_feature2,test_target2))  # Try to attain a positive value
from sklearn.model_selection import ParameterGrid
grid = {'n_estimators': [1000], 'max_depth': [500], 'max_features': [5], 'random_state': [7]}
test_scores = []
# Loop through the parameter grid, set the hyperparameters, and save the scores
for g in ParameterGrid(grid):
    rfr.set_params(**g)  # ** is "unpacking" the dictionary
    rfr.fit(train_feature2, train_target2)
    test_scores.append(rfr.score(test_feature2, test_target2))
best_idx = np.argmax(test_scores)
print(test_scores[best_idx], ParameterGrid(grid)[best_idx])  # You don't want negative value
rfr = RandomForestRegressor(n_estimators=1000, max_depth=500, max_features = 5, random_state=7)
rfr.fit(train_feature2, train_target2)
pickle.dump(rfr, open('model.pkl','wb'))# Saving model to disk
model = pickle.load(open('model.pkl','rb'))# Loading model to compare the results
#print(model.predict([[3,37,25,32,74,5.4,10]]))