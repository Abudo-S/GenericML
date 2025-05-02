import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import itertools
from collections import Counter
import pandas as pd
from scipy.stats import entropy

class Node:
    def __init__(self, feature=None, feature_value_threshold=None, left=None, right=None, value = None):
        self.feature = feature
        self.feature_value_threshold = feature_value_threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None
    
class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        #expand the tree
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.root = self._expand_tree(X, y)

    def _expand_tree(self, X:np.ndarray, y: np.ndarray, depth=0):
        n_samples, n_features = X.shape #2D
        n_labels = len(np.unique(y))

        #stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            #most common label within the node (leaf nodes shouldn't have more than 2 labels)
            leaf_node = Counter(y).most_common(1)[0][0] if n_labels > 0 else None 
            return Node(value=leaf_node)
        
        feature_indices = np.random.choice(n_features, self.n_features, replace=False) #get random features of size n_features
        best_feat, best_value_threshold = self._choose_split_criteria_greedy(X, y, feature_indices)
        left_indices, right_indices = self._split(X[:, best_feat], best_value_threshold)
        left = self._expand_tree(X[left_indices], y[left_indices], depth+1)
        right = self._expand_tree(X[right_indices], y[right_indices], depth+1)

        return Node(best_feat, best_value_threshold, left, right)

    def _choose_split_criteria_greedy(self, X, y, feature_indices):
        best_gain = -1
        split_feature_idx, split_feature_value= None, None

        for feature_index in feature_indices:
            X_column = X[:, feature_index]
            feature_values = np.unique(X_column) #get unique values from a feature column

            for feature_value in feature_values:
                #calculate the information gain for each feature's value (then use the highest info gain as a split threshold)
                value_info_gain = self._calculate_feature_info_gain(y, X_column, feature_value)

                if value_info_gain > best_gain:
                    best_gain = value_info_gain
                    split_feature_idx = feature_index
                    split_feature_value = feature_value

        return split_feature_idx, split_feature_value

    '''
    another split criteria is the linear "gini function"
    '''
    def _calculate_feature_info_gain(self, y, X_column, feature_value):
        parent_entropy = entropy(y)

        left_indices, right_indices = self._split(X_column, feature_value)

        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0
        
        child_entropy = (len(left_indices) / len(y)) * entropy(y[left_indices]) + (len(right_indices) / len(y)) * entropy(y[right_indices])

        return parent_entropy - child_entropy
    
    def _split(self, X_column, feature_value):
        left_indices = np.argwhere(X_column <= feature_value).flatten()
        right_indices = np.argwhere(X_column > feature_value).flatten()

        return left_indices, right_indices

    def predict(self, X: np.ndarray):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node:Node):
        if node.is_leaf():
            return node.value
        
        return self._traverse_tree(x, node.left) if x[node.feature] <= node.feature_value_threshold else self._traverse_tree(x, node.right)
    


dataset = datasets.load_iris()
X = dataset.data #for each record: vector of features of data point x
y = dataset.target #for each record: label associated to x

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=1234)

#print(f'shape {X_train.shape}\n', X_train) #4D dataset
#print(y)
#print(X[:,0])
#print(X[:,1])

plt.figure()
plt.scatter(X[:,0], X[:,1], c=y, edgecolors='k', s=20) #for simplicity of visualization, we'll consider the first column as x, and the second column as y 
#plt.show()


decisionTree = DecisionTree()
decisionTree.fit(X_train, y_train)
predictions = decisionTree.predict(X_test)
accuracy = np.sum(predictions == y_test) / len(y_test)
print(f'Acc.:{accuracy}') #for classification problems we use the accuracy
