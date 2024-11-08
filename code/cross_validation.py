import numpy as np


def cross_validation_score(model, X, y, folds, metric):
    """run cross validation on X and y with specific model by given folds. Evaluate by given metric."""
    metric_value = []
    for train_indices, validation_indices in folds.split(X):
        X_test = X[validation_indices]
        X_train = X[train_indices]
        y_train = y[train_indices]
        y_test = y[validation_indices]
        model.fit(X_train, y_train)
        y_pred = np.array(model.predict(X_test))
        metric_value.append(metric(y_test, y_pred))
    return metric_value


def model_selection_cross_validation(model, k_list, X, y, folds, metric):
    """ run cross validation on X and y for every model induced by values from k_list by given folds.
        Evaluate each model by given metric."""
    mean_list = []
    std_list = []
    for k in k_list:
        knn = model(k=k)
        scores = np.array(cross_validation_score(knn, X, y, folds, metric))
        mean_list.append(np.mean(scores))
        std_list.append(np.std(scores, ddof=1))
    return mean_list, std_list


