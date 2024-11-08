import numpy as np
import matplotlib.pyplot as plt


def f1_score(y_true, y_pred):
    """returns f1_score of binary classification task with true labels y_true and predicted labels y_pred"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    return (2 * recall * precision) / (recall + precision)


def rmse(y_true, y_pred):
    """returns RMSE of regression task with true labels y_true and predicted labels y_pred"""
    n = y_true.shape[0]
    sum = np.sum((y_true - y_pred) ** 2)
    return np.sqrt(sum / n)


def visualize_results(k_list, scores, metric_name, title, path):
    """plot a results graph of cross validation scores"""
    plt.figure(figsize=(8, 8))
    plt.xlabel('k', fontsize=15)
    plt.ylabel(metric_name, fontsize=15)
    plt.title(title, fontsize=20)
    plt.plot(k_list, scores, color='blue')
    plt.savefig(path.format(2))
