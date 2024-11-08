# K-Nearest Neighbors (KNN) Model Project

This project implements a customizable K-Nearest Neighbors (KNN) model for both classification and regression tasks. Using a dataset with weather and season-related features, it provides a full pipeline for data preprocessing, cross-validation, and model evaluation. Key components include modules for data loading, KNN model definition, cross-validation, and performance visualization.

## Project Structure

### `knn.py`: KNN Model Classes
- **`KNN` (base class)**: Implements core functions such as calculating Euclidean distances and finding nearest neighbors.
- **`ClassificationKNN`**: Inherits from KNN, implementing classification by selecting the majority label among the k-nearest neighbors.
- **`RegressionKNN`**: Inherits from KNN, implementing regression by averaging the target values of the k-nearest neighbors.

### `data.py`: Data Management Functions
- **`load_data()`**: Loads the CSV dataset.
- **`adjust_labels()`**: Adjusts season labels to binary categories.
- **`add_noise()`**: Adds Gaussian noise to the dataset to improve model robustness.
- **`StandardScaler`**: Class for scaling features by mean and standard deviation.
- **`get_folds()`**: Defines the cross-validation folds.

### `cross_validation.py`: Cross-Validation Functions
- **`cross_validation_score()`**: Computes model performance on each fold of the dataset.
- **`model_selection_cross_validation()`**: Selects the optimal k value by running cross-validation for different values of k.

### `evaluation.py`: Performance Metrics and Visualization
- **`f1_score()`**: Calculates the F1 score for binary classification.
- **`rmse()`**: Calculates the Root Mean Squared Error (RMSE) for regression.
- **`visualize_results()`**: Plots cross-validation scores for different k values.

### `main.py`: Main Script
- **Loads the dataset**.
- **Runs cross-validation** with various k values for both classification and regression tasks.
- **Visualizes and saves performance metrics**.

---

### Note
When running `main.py`, replace `<path_to_csv_file>` with the full path to the London bike rentals dataset CSV file on your system.
