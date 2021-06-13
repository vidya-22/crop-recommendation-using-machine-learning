# Importing the libraries
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('data.csv')
#LOGISTIC REGRESSION


y = data['label']
x = data.drop(['label'], axis = 1)

# lets split the Dataset for Predictive Modelling
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
print('Observations for Target: 2200')
print('Training Observations for Target: 1650')
print('Testing Observations for Target: 550')
# Create the logistic regression model and fit to the training data
lr= LogisticRegression()
lr.fit(x_train, y_train)
#Fitting model with trainig data


# Saving model to disk
pickle.dump(lr, open('logisticmodel.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('logisticmodel.pkl','rb'))

