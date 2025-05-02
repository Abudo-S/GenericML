import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import itertools
from collections import Counter
import pandas as pd

class KNN:
    def __init__(self, k=3):
        self.k = k

    '''
    Note that there's no training phase in KNN, it's just storing data
    '''
    def fit(self, X, y):
        self.X__training_set = X
        self.y_training_set = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    '''
    steps:
    -compute the distances between x and all other instances in X__training_set
    -choose the nearest k neighboring instances 
    -return the major label in k instances [in case of pair k, break the equality by retrieving the nearst neigbors]
    '''
    def _predict(self, x_to_predict):
        X_distances = [np.sqrt(np.sum((x_to_predict - x)**2)) for x in self.X__training_set]
        k_X_nearest_indices = np.argsort(X_distances)[:self.k]
        k_X_nearest_labels = [self.y_training_set[idx] for idx in k_X_nearest_indices]

        #most_common(n) returns [(y, y_count), ..., n]
        #return Counter(k_X_nearest_labels).most_common(1)[0][0]

        k_X_nearest_labels_dist = {(self.y_training_set[idx], X_distances[idx]) for idx in k_X_nearest_indices} #already sorted by distances
        k_X_nearest_labels_dist_groups = itertools.groupby(k_X_nearest_labels_dist, key=lambda kvp: (kvp[0], np.sum(kvp[1])))
        predicted_y = sorted([*k_X_nearest_labels_dist_groups], key=lambda group: (len(list(group[1])), np.min(group[0][1])))[0][0][0]
        return predicted_y


if __name__ == "__main__":
    iris_dataset = datasets.load_iris()
    X = iris_dataset.data #for each record: vector of features of data point x
    y = iris_dataset.target #for each record: label associated to x

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=1234)

    #print(f'shape {X_train.shape}\n', X_train) #4D dataset
    #print(y)
    print(X[:,0])
    print(X[:,1])

    plt.figure()
    plt.scatter(X[:,0], X[:,1], c=y, edgecolors='k', s=20) #for simplicity of visualiztion, we'll consider the first column as x, and the second column as y 
    #plt.show()

    knn = KNN(3)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    accuracy = np.sum(predictions == y_test) / len(y_test)
    print(f'Acc.:{accuracy}') #for classification problems we use the accuracy