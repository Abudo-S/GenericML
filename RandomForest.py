import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from collections import Counter
from DecisionTree import DecisionTree

'''
The power of Random Forest comes from the combination of these "random" elements:

-Reduced Variance (due to Bagging): By training multiple trees on different bootstrap samples and then averaging (or taking the majority vote) 
their predictions, Random Forest reduces the variance of the individual trees.

Reduced Correlation (due to Feature Randomness): By randomly selecting a subset of features at each split, the trees in the forest become more decorrelated. 
If all trees were built using the same strong predictor features, their predictions would be highly correlated, 
and the benefit of ensembling would be limited.
'''
class RandomForest:
    trees:list[DecisionTree]

    def __init__(self, n_trees=100, min_samples_split=2, max_depth=100, n_feats=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats

    def fit(self, X, y):
        self.trees = []

        for _ in range(self.n_trees):
            tree = DecisionTree(self.min_samples_split, self.max_depth, self.n_feats)
            X_sample, y_sample = self.bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        treesPreds = np.array([tree.predict(X) for tree in self.trees]) #[predTree1[] predTree2[] ...]
        treesPreds = np.swapaxes(treesPreds, 0, 1)
        return np.array([most_common_label(treePred) for treePred in treesPreds])

    '''
    Random Forests introduce randomness in two key ways when building each individual decision tree:

    -Bootstrap Sampling (Bagging): Instead of using the entire training dataset to build each tree, 
    Random Forest creates multiple subsets of the training data by sampling with replacement.
    This means that some original data points might be included in multiple subsets, while others might be left out.
    Each of these subsets is used to train a separate decision tree. This process of sampling with replacement is called bootstrapping,
    and combining the results of multiple models trained on these bootstrap samples is called bagging.   

    -Random Feature Subselection (Feature Randomness): When splitting a node in a decision tree, 
    instead of considering all possible features to find the best split, Random Forest randomly selects a subset of features 
    (typically a small fraction of the total number of features). The best split is then chosen only from this random subset of features.
    This forces each tree to consider a different set of features, leading to more diverse trees.
    '''
    def bootstrap_sample(self, X: np.ndarray, y: np.ndarray):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, size= n_samples, replace=True)
        return X[idxs], y[idxs]

def most_common_label(y:np.ndarray):
    return Counter(y).most_common(1)[0][0]

if __name__ == "__main__":
    iris_dataset = datasets.load_iris()
    X = iris_dataset.data #for each record: vector of features of data point x
    y = iris_dataset.target #for each record: label associated to x

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=1234)

    #print(X)
    print(f'shape {X_train.shape}\n', X_train) #4D dataset
    #print(y)
    print(X[:,0])
    print(X[:,1])

    plt.figure()
    plt.scatter(X[:,0], X[:,1], c=y, edgecolors='k', s=20) #for simplicity of visualiztion, we'll consider the first column as x, and the second column as y 
    #plt.show()

    rf = RandomForest(3)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    accuracy = np.sum(predictions == y_test) / len(y_test)
    print(f'Acc.:{accuracy}') #for classification problems we use the accuracy
    #in case of base of bad accuracy we might try to use preprocessing techniques like "scale standarization"
