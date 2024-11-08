import sys
import numpy as np
import evaluation
import knn
import data
import cross_validation


def main(argv):
    df = data.load_data((argv[1]))
    data.adjust_labels(df['season'])
    k_list = [3, 5, 11, 25, 51, 75, 101]
    folds = data.get_folds()
    y_train = np.array(df['season'])
    X_train = data.add_noise(np.array(df.loc[:, ["t1", "t2", "wind_speed", "hum"]]))
    print("Part 1 – Classification")
    mean, std = cross_validation.model_selection_cross_validation(knn.ClassificationKNN, k_list, X_train, y_train,
                                                                  folds, evaluation.f1_score)
    for k in range(len(k_list)):
        print(f"k={k_list[k]}, mean score: {format(mean[k], '.4f')}, std of scores: {format(std[k], '.4f')}")
    path2 = '/home/studentHW3/plot{}.png'

    evaluation.visualize_results(k_list, mean, "f1_score", "Classification", path2)
    print()
    print("Part2 - Regression")
    x_train2 = data.add_noise(np.array(df.loc[:, ["t1", "t2", "wind_speed"]]))
    y_train2 = np.array(df.loc[:, "hum"])
    mean2, std2 = cross_validation.model_selection_cross_validation(knn.RegressionKNN, k_list, x_train2, y_train2,
                                                                    folds, evaluation.rmse)
    for k in range(len(k_list)):
        print(f"k={k_list[k]}, mean score: {format(mean2[k], '.4f')}, std of scores: {format(std2[k], '.4f')}")
    evaluation.visualize_results(k_list, mean2, "RMSE", "Regression", path2)


if __name__ == '__main__':
    main(sys.argv)
