import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
np.random.seed(2)


def load_data(path):
    """reads and returns the pandas DataFrame"""
    df = pd.read_csv(path)
    return df


def adjust_labels(y):
    """adjust labels of season from {0,1,2,3} to {0,1}"""
    return y.replace({1: 0, 2: 1, 3: 1}, inplace=True)


def add_noise(data):
    """
    :param data: dataset as np.array of shape (n, d) with n observations and d features
    :return: data + noise, where noise~N(0,0.001^2)
    """
    noise = np.random.normal(loc=0, scale=0.001, size=data.shape)
    return data + noise


def get_folds():
    """
    :return: sklearn KFold object that defines a specific partition to folds
    """
    return KFold(n_splits=5, shuffle=True, random_state=34)


class StandardScaler:
    def __init__(self):
        """object instantiation"""
        self.mean = None
        self.std = None

    def fit(self, X):
        """fit scaler by learning mean and standard deviation per feature"""
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0, ddof=1)

    def transform(self, X):
        """transform X by learned mean and standard deviation, and return it"""
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        """fit scaler by learning mean and standard deviation per feature, and then transform X"""
        StandardScaler.fit(self, X)
        return StandardScaler.transform(self, X)
