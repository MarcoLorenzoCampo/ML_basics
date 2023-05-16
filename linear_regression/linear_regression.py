import pickle

import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd
import sklearn
from matplotlib import style

data = pd.read_csv('student-mat.csv', sep=';')

data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]

predict = 'G3'

# Training data is the original reduced dataset without grade 3 (G3)
# that we want to predict
x = np.array(data.drop([predict], 1))

# Target variable is G3 that we want to predict
y = np.array(data[predict])

# Splitting the attributes and target at random, keeping 90% to train, 10% to test proportions
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# Just running the training process for an arbitrary amount of times
# and picking the best accuracy possible
'''
best = 0
for _ in range(30):
    # IMPORTANT: This split is in the for loop since it provides a random
    # split of data, yielding different accuracy results each time
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    # Linear regression
    linear = linear_model.LinearRegression()

    # Fit the data to find the best fit 'line'
    linear.fit(x_train, y_train)
    accuracy = linear.score(x_test, y_test)

    # Coefficients are 5 since we have 5 attributes, we need 5 slope
    # To determine the hyperplane that best fits
    print('Accuracy: ', accuracy)
    print('Coefficients: ', linear.coef_)
    print('Intercept: ', linear.intercept_, '\n')

    if accuracy > best:
        best = accuracy
        # The accuracy varies a lot 66% -> 89%, we might need to save the best one
        with open('student_model.pickle', 'wb') as f:
            pickle.dump(linear, f)

print('Best Accuracy: ', best)
'''

# Commenting the training part when we have a pretty accurate model saved
pickle_in = open('student_model.pickle', "rb")
linear = pickle.load(pickle_in)

predictions = linear.predict(x_test)

# Now for all the predictions, we print:
# The predicted grade, the attributes used for prediction
# and the actual value we wanted to predict
for x in range(len(predictions)):
    print('prediction: ', predictions[x], '| Attributes: ', x_test[x], '| Actual G3: ', y_test[x])

# Plotting the results, p is an arbitrary attribute
p = 'failures'
style.use('ggplot')
pyplot.scatter(data[p], data['G3'])
pyplot.xlabel(p)
pyplot.ylabel("Final grade")
pyplot.show()
