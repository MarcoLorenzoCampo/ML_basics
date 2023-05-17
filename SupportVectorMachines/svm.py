import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

# Loading the breast cancer dataset from sklearn
cancer = datasets.load_breast_cancer()

# The data features are known, the target is weather the cancer is malign or not
x = cancer.data
y = cancer.target

# Splitting the attributes and target at random, keeping 80% to train, 20% to test proportions
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

classes = ['malignant', 'benign']

# build a classifier. SVC = Support Vector Classification
# kernels: linear, poly (set degree=..), sigmoid..
# C is the soft margin level
clf_svm = svm.SVC(kernel='linear', C=2)

# just for comparison using the KNN classifier
clf_KNN = KNeighborsClassifier(n_neighbors=9)
clf_KNN.fit(x_train, y_train)

# train the classifier providing input and output in order to build the classification criterion
clf_svm.fit(x_train, y_train)


# make a prediction on the test set
y_prediction = clf_svm.predict(x_test)

# measure the accuracy of the prediction with the output test set
acc_svm = metrics.accuracy_score(y_test, y_prediction)

# accuracy of the knn classifier
acc_knn = clf_KNN.score(x_test, y_test)

print('SVM: ', acc_svm, '| KNN: ', acc_knn)