import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
data = pd.read_csv('data.csv')
crop=data.values
X3= crop[:,0:-1]  # slice all rows and start with column 0 and go up to but not including the last column
y3 = crop[:,-1]  # slice all rows and last column, essentially separating out 'crop' column
Y3=y3
le = preprocessing.LabelEncoder()
if Y3.dtype == object:
    Y3 = le.fit_transform(Y3)
else:
    pass
train_feature2, test_feature2, train_target2, test_target2 = train_test_split(X3, Y3, test_size=0.25, random_state=1)
print('Observations for Target: %d' % (len(Y3)))
print('Training Observations for Target: %d' % (len(train_target2)))
print('Testing Observations for Target: %d' % (len(test_target2)))
# Create the gradient boosting model and fit to the training data
gbr = GradientBoostingRegressor(max_features=5, learning_rate=0.01, n_estimators=1000, max_depth=500, subsample=1, random_state=1000)
gbr.fit(train_feature2, train_target2)
print(gbr.score(train_feature2, train_target2))
print(gbr.score(test_feature2, test_target2))
from sklearn.model_selection import ParameterGrid
grid = {'n_estimators': [1000], 'max_depth': [500], 'max_features': [5], 'random_state': [1000], 'learning_rate':[0.01], 'subsample':[1]}
test_scores = []
for g in ParameterGrid(grid):# Loop through the parameter grid, set the hyperparameters, and save the scores
    gbr.set_params(**g)  # ** is "unpacking" the dictionary
    gbr.fit(train_feature2, train_target2)
    test_scores.append(gbr.score(test_feature2, test_target2))
best_idx = np.argmax(test_scores)# Find best hyperparameters from the test score and print
print(test_scores[best_idx], ParameterGrid(grid)[best_idx])  # You don't want negative value
gbr = GradientBoostingRegressor(max_features=5, learning_rate=0.01, n_estimators=1000, max_depth=500, subsample=1, random_state=1000)
gbr.fit(train_feature2, train_target2)
pickle.dump(gbr, open('gradientmodel.pkl','wb'))
model = pickle.load(open('gradientmodel.pkl','rb'))# Loading model to compare the results
#print(model.predict([[3,37,25,32,74,5.4,10]]))