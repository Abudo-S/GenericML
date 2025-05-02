import numpy as np
import random

'''
Simple hyperparameter tuning based on evaluating (per k iterations) a predictor objective_function
'''
def hyperparameter_tuning(objective_function, param_grid, num_iterations=10, random_seed=None):
    """
    Performs random search for hyperparameter tuning.

    Args:
        objective_function: A function that takes a dictionary of hyperparameters
                              and returns a score (e.g., validation accuracy).
        param_grid: A dictionary where keys are hyperparameter names and values are
                     lists of possible values.
        num_iterations: The number of random hyperparameter combinations to try.
        random_seed: Optional seed for the random number generator for reproducibility.

    Returns:
        A dictionary containing the best hyperparameters and their corresponding score.
    """

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    best_score = float('-inf')  # Initialize with negative infinity
    best_params = None

    for _ in range(num_iterations):
        # Generate random hyperparameter combination
        params = {}
        for param_name, param_values in param_grid.items():
            params[param_name] = random.choice(param_values)

        # Evaluate the objective function
        score = objective_function(params)

        # Update best parameters if a better score is found
        if score > best_score:
            best_score = score
            best_params = params

    return {'best_params': best_params, 'best_score': best_score}

# Example usage (replace with your actual objective function and parameter grid)
def example_objective_function(params):
    """
    A simple example objective function. Replace with your actual model evaluation.
    """
    # Simulate a model's performance based on hyperparameters
    learning_rate = params['learning_rate']
    num_units = params['num_units']
    activation = params['activation']

    # Example calculation (replace with your model's evaluation logic)
    score = -abs(learning_rate - 0.01) + num_units / 100.0
    if activation == 'relu':
        score += 0.1
    return score

# Example parameter grid
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1, 0.2],
    'num_units': [32, 64, 128, 256],
    'activation': ['relu', 'tanh', 'sigmoid']
}

# Run hyperparameter tuning
results = hyperparameter_tuning(example_objective_function, param_grid, num_iterations=50, random_seed=42)

# Print results
print("Best Hyperparameters:", results['best_params'])
print("Best Score:", results['best_score'])

# Example with a more complex objective function, using numpy arrays.

def example_numpy_objective(params):
    """
    Example objective function utilizing numpy arrays.
    """
    learning_rate = params['learning_rate']
    regularization = params['regularization']
    data_size = params['data_size']

    # Simulate a model with numpy operations.
    X = np.random.rand(data_size, 10)
    y = np.random.rand(data_size)
    model_output = np.dot(X, np.random.rand(10)) * learning_rate
    loss = np.mean((model_output - y)**2) + regularization * np.sum(np.abs(np.random.rand(10)))

    return -loss # minimize loss, maximize negative loss.

complex_param_grid = {
    'learning_rate': np.logspace(-4, -1, 50).tolist(),
    'regularization': np.logspace(-6, -2, 50).tolist(),
    'data_size': [100, 500, 1000, 2000]
}

complex_results = hyperparameter_tuning(example_numpy_objective, complex_param_grid, num_iterations = 100, random_seed = 123)

print("\nComplex best params:", complex_results['best_params'])
print("Complex best score:", complex_results['best_score'])