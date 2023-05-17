"""
KNN (K Nearest Neighbors algorithm) for Classification
If we chose K=3, then we look at the 3 closest points of our dataset
to the point we want to classify. The new point belongs to the class
that has the highest number of closest points.
K should be odd to avoid ties.
"""

import sklearn
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing

data = pd.read_csv('car.data')

# Don't use non numerical data, all of our data is non numerical
# Convert attributes to numerical data -> preprocessing from sklearn

# Create a LabelEncoder, it encodes non numerical labels into numerical
le = preprocessing.LabelEncoder()

# Almost all the labels are non-numerical
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = 'class'

# Zip created a tuple from different objects, so we have a list of tuples
x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

# Splitting the attributes and target at random, keeping 90% to train, 10% to test proportions
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# Create classifier, K=9, change the parameter based on the output accuracy
K = 9
model = KNeighborsClassifier(K)
model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print(acc)

# Predicting the classification of the test data
predicted = model.predict(x_test)

names = ['unacc', 'acc', 'good', 'vgood']
for x in range(len(x_test)):
    print('Predicted data: ', names[predicted[x]], '| Data: ', x_test[x], '| Actual: ', y_test[x])

    # Print the distance to all the other points
    n = model.kneighbors([x_test[x]], K, True)
    print('N: ', n)

"""
Let's take a random output for example:

Predicted data:  good | Data:  (1, 1, 3, 2, 0, 1) | Actual:  2
N:  (
    array([[1, 1, 1, 1, 1, 1, 1, 1, 1.41421356]]), 
    array([[ 315,  750, 1094, 1205,  310,  491,  605, 1335,  349]],
      dtype=int64)
    )

Our random element of the test set has distance 1 from K=1-8 but has distance
1.41421356 for the the 9th closest element.
The second array gives us the index number of the 9 nearest neighbors
"""