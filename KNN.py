import sys

import numpy as np
from utility import euclidean_distance
from collections import Counter


class KNN:

    def __init__(self, k=3):
        self.k = k
        self.X_train = []
        self.y_train = []

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _predict(self, x):
        # Get all distances between x sample and other training samples
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # Get labels of k nearest samples
        k_nearest_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]

        # Get the most majority label
        most_common = Counter(k_nearest_labels).most_common(1)
        prediction_label = most_common[0][0]

        return prediction_label

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]

        return np.array(predicted_labels)

    def accuracy(self, y_hat, Y):
        return np.sum(y_hat == Y) / len(Y)
