import numpy as np
import random
from sklearn.model_selection import KFold
from KNN import KNN
from sklearn import datasets
from sklearn.model_selection import train_test_split

def nested_cross_validation(model_function, param_grid, X, y, inner_cv=3, outer_cv=5, num_iterations=10, random_seed=None):
    """
    Performs nested cross-validation for hyperparameter tuning and model evaluation.

    Args:
        model_function: A function that takes hyperparameters, X_train, y_train, X_val, y_val, X_test, y_test
                          and returns a score (e.g., test set accuracy).
        param_grid: A dictionary where keys are hyperparameter names and values are
                     lists of possible values.
        X: The feature matrix.
        y: The target vector.
        inner_cv: The number of folds for the inner cross-validation loop (hyperparameter tuning).
        outer_cv: The number of folds for the outer cross-validation loop (model evaluation).
        num_iterations: The number of random hyperparameter combinations to try in the inner loop.
        random_seed: Optional seed for the random number generator for reproducibility.

    Returns:
        A list of scores obtained from each outer fold.
    """

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    outer_kf = KFold(n_splits=outer_cv, shuffle=True, random_state=random_seed)
    outer_scores = {}

    for outer_train_index, outer_test_index in outer_kf.split(X, y):
        X_outer_train, X_outer_test = X[outer_train_index], X[outer_test_index]
        y_outer_train, y_outer_test = y[outer_train_index], y[outer_test_index]

        # Inner cross-validation loop for hyperparameter tuning
        inner_kf = KFold(n_splits=inner_cv, shuffle=True, random_state=random_seed)
        best_inner_score = float('-inf')
        best_inner_params = None

        for _ in range(num_iterations):
            params = {}
            for param_name, param_values in param_grid.items():
                params[param_name] = random.choice(param_values)

            inner_scores = []
            for inner_train_index, inner_val_index in inner_kf.split(X_outer_train, y_outer_train):
                X_inner_train, X_inner_val = X_outer_train[inner_train_index], X_outer_train[inner_val_index]
                y_inner_train, y_inner_val = y_outer_train[inner_train_index], y_outer_train[inner_val_index]

                inner_score = model_function(params, X_inner_train, y_inner_train, X_inner_val, y_inner_val) #Don't pass test data to inner loop.
                inner_scores.append(inner_score)

            avg_inner_score = np.mean(inner_scores)

            if avg_inner_score > best_inner_score:
                best_inner_score = avg_inner_score
                best_inner_params = params

        # Evaluate the model with the best hyperparameters on the outer test set
        outer_scores['@'.join([f'{k}:{v}' for k,v in best_inner_params.items()])]= model_function(best_inner_params, X_outer_train, y_outer_train, X_outer_test, y_outer_test) #Pass test data to outer loop.

    return outer_scores

#Ex. knn
def model_function(params, X_train, y_train, X_test, y_test):
    k = params['k']

    knn = KNN(k)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    return np.sum(predictions == y_test) / len(y_test) #accurcy

# # Example data
# X = np.random.rand(100, 10)
# y = np.random.rand(100)

# # Example parameter grid
# param_grid = {
#     'learning_rate': [0.001, 0.01, 0.1],
#     'num_units': [32, 64, 128]
# }

param_grid = {
    'k': [1, 3, 5, 7, 9, 11]
}

iris_dataset = datasets.load_iris()
X = iris_dataset.data #for each record: vector of features of data point x
y = iris_dataset.target #for each record: label associated to x

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=1234)

# Run nested cross-validation
outer_scores = nested_cross_validation(model_function, param_grid, X, y, inner_cv=3, outer_cv=5, num_iterations=5, random_seed=42)

# Print results
print("Outer Fold Scores:", outer_scores)
#print("Average Outer Score:", np.mean(outer_scores.values()))