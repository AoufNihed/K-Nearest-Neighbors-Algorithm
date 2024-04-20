import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum(x1 - x2) ** 2)
    return distance

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, Y):
        self.X_tain = X
        self.y_train = Y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # calculate the distance
        distances = [euclidean_distance(x, self.X_tain[i]) for i in range(len(self.X_tain))]

        # get the closest K
        K_indices = np.argsort(distances)[:self.k]
        K_nearest_labels = [self.y_train[i] for i in K_indices]

        # get the majority vote
        most_common = Counter(K_nearest_labels).most_common()
        return most_common