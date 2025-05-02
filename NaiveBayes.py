import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

'''
Bayes theorem P(y|X) = (P(X|y) * P(y)) / P(X) = (P(x1|y) * P(xn|y) * P(y)) / P(X)
predictor formula y = argMax_y (log(P(x1|y)) + log(P(xn|y)) + log(P(y)))
conditional prob. P(xi|y) = (1/sqrt(2.π.σ²)) . exp
Naive bayes doesn't exploit the relationship between features
'''
class NaiveBayes:
    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for c in self.classes:
            X_c = X[c==y]
            self._mean[c, :] = X_c.mean(axis=0) #mean of each instance
            self._var[c, :] = X_c.var(axis=0) #variance of each instance
            self._priors[c] = X_c.shape[0] / float(n_samples) #he prior probability represents the initial belief or knowledge about the probability of each class occurring before observing 

    def predict(self, X):
        return [self._predict(x) for x in X]

    #calculate the max of posterior probs
    def _predict(self, x):
        posteriors = [np.log(self._priors[idx]) + np.sum(np.log(self._probDensityFunction(idx, c))) for idx, c in enumerate(self.classes)]

        return self.classes[np.argmax(posteriors)]
    
    def _probDensityFunction(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        return np.exp(- (x - mean)**2 / (2 * var)) / np.sqrt(2 * np.pi * var)
    
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

    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)
    accuracy = np.sum(predictions == y_test) / len(y_test)
    print(f'Acc.:{accuracy}') #for classification problems we use the accuracy
    #in case of base of bad accuracy we might try to use preprocessing techniques like "scale standarization"