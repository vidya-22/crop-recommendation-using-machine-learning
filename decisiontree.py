import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor
data = pd.read_csv('data.csv')
crop=data.values
X3= crop[:,0:-1]  # slice all rows and start with column 0 and go up to but not including the last column
y3 = crop[:,-1]  # slice all rows and last column, essentially separating out 'crop' column
Y3=y3
le = preprocessing.LabelEncoder()# converting crop column to numeric values
if Y3.dtype == object:#for column_name in tes_data.columns:
    Y3 = le.fit_transform(Y3)
else:
    pass
train_feature2, test_feature2, train_target2, test_target2 = train_test_split(X3, Y3, test_size=0.25, random_state=1)
print('Observations for Target: %d' % (len(Y3)))
print('Training Observations for Target: %d' % (len(train_target2)))
print('Testing Observations for Target: %d' % (len(test_target2)))
# Create the decision tree model and fit to the training data
dr=DecisionTreeRegressor()
from sklearn.model_selection import ParameterGrid
grid = { 'max_depth': [100], 'max_features': [5], 'random_state': [50]}
test_scores = []
for g in ParameterGrid(grid):# Loop through the parameter grid, set the hyperparameters, and save the scores
    dr.set_params(**g)  # ** is "unpacking" the dictionary
    dr.fit(train_feature2, train_target2)
    test_scores.append(dr.score(test_feature2, test_target2))
best_idx = np.argmax(test_scores)# Find best hyperparameters from the test score and print
print(test_scores[best_idx], ParameterGrid(grid)[best_idx])  # You don't want negative value
dr=DecisionTreeRegressor(max_depth=100, max_features=5, random_state=50)
dr.fit(train_feature2, train_target2)
print(dr.score(train_feature2, train_target2))
print(dr.score(test_feature2,test_target2)) 
pickle.dump(dr, open('decisiontreemodel.pkl','wb'))


# Loading model to compare the results
model = pickle.load(open('decisiontreemodel.pkl','rb'))
