import numpy as np
import sklearn
from sklearn import metrics
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans

# Load data
digits = load_digits()

# Scale all the features from digits so that they are [-1,1]
data = scale(digits.data)

# Get labels
y = digits.target

# Number of clusters obtained dynamically
k = len(np.unique(y))
samples, features = data.shape


# Function to score the model from sklearn library
def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y, estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))

# Building the classifier that uses the KMeans
# n_clusters = number of clusters required
# init = strategy to initialize the centroids
# n_init = number of times algo will run with different seeds
# max_iterations = 300 default value
clf = KMeans(n_clusters=k, init='k-means++', n_init=10)

# Measure accuracy
bench_k_means(clf, "Classifier 1:  ", data)