import numpy as np
from scipy import stats
from abc import abstractmethod
from data import StandardScaler


class KNN:
    def __init__(self, k):
        """object instantiation, save k and define a scaler object"""
        self.k = k
        self.X = None
        self.y = None
        self.scaler = StandardScaler()

    def fit(self, X_train, y_train):
        """fit scaler and save X_train and y_train"""
        self.y = y_train
        self.X = self.scaler.fit_transform(X_train)

    @abstractmethod
    def predict(self, X_test):
        """predict labels for X_test and return predicted labels"""

    def neighbours_indices(self, x):
        """for a given point x, find indices of k closest points in the training set """
        distances = {}
        length = len(self.X)
        for i in range(length):
            sub_distance = KNN.dist(x, self.X[i])
            distances[i] = sub_distance
        sorted_distances = (sorted(distances.items(), key=lambda value: value[1]))
        neighbours = []
        for i in range(self.k):
            neighbours.append(sorted_distances[i][0])
        return neighbours

    @staticmethod
    def dist(x1, x2):
        """ returns Euclidean distance between x1 and x2 """
        return np.sqrt(np.sum((x1 - x2)**2))


class ClassificationKNN(KNN):
    def __init__(self, k):
        """ object instantiation, parent class instantiation """
        super().__init__(k)

    def predict(self, X_test):
        """ predict labels for X_test and return predicted labels """
        predicted_labels = []
        X_test = self.scaler.transform(X_test)
        for x in X_test:
            neighbors = KNN.neighbours_indices(self, x)
            k_nearest_labels = [self.y[i] for i in neighbors]
            predicted_labels.append(stats.mode(k_nearest_labels, axis=None)[0][0])

        return predicted_labels


class RegressionKNN(KNN):
    def __init__(self, k):
        """ object instantiation, parent class instantiation """
        super().__init__(k)

    def predict(self, X_test):
        """ predict labels for X_test and return predicted labels """
        predicted_labels = []
        X_test = self.scaler.transform(X_test)
        for x in X_test:
            neighbors = KNN.neighbours_indices(self, x)
            k_nearest_labels = [self.y[i] for i in neighbors]
            predicted_labels.append(np.array(k_nearest_labels).mean())
        return predicted_labels
