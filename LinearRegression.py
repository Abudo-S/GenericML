import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import itertools
from collections import Counter
import pandas as pd

'''
it seeks to find a linear equation that best describes how one or more independent variables (predictors, features)
relate to a dependent variable (target).
The model aims to find the "best-fit" line or hyperplane that MINIMIZES (not exact value predictor)
the difference between the predicted values and the actual values of the dependent variable
'''
class LinearRegression:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.bias = None
        self.weights = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        print(type(X))
        n_sample, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias

            first_derivative_dw = (1 / n_sample) * np.dot(X.T, (y_predicted - y))
            second_derivative_db = (1 / n_sample) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * first_derivative_dw #adjust weights with respect to the learning rate
            self.bias -= self.learning_rate * second_derivative_db #adjust bias with respect to the learning rate

    def predict(self, X: np.ndarray):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted

def mean_sqr_error(y_true, y_predicted):
    return np.mean((y_true - y_predicted)**2)

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=1234)

#print(f'shape {X_train.shape}\n', X_train) #1D dataset
print(y)

plt.figure()
plt.scatter(X[:,0], y,marker='o', edgecolors='k', s=30) #for simplicity of visualiztion, we'll consider the first column as x, and the second column as y 
#plt.show()

linRegression = LinearRegression(learning_rate=0.01)
linRegression.fit(X_train, y_train)
prediction_line = linRegression.predict(X_test)
print(f'Sqr.err.:{mean_sqr_error(y_test, prediction_line)}') #for regression problems we use the mean square root

plt.close()
plt.figure(figsize=(8,6))
cmap = plt.get_cmap()
plt.scatter(X_train, y_train,marker='o', color=cmap(0.5), s=10)
plt.scatter(X_test, y_test,marker='o', color=cmap(0.9), s=10)
plt.plot(X_test, prediction_line, color='black', linewidth=2, label='prediction')
plt.show()
