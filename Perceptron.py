import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

'''
Perceptron is used in supervised learning for binary classification, it's considered as the simplest unit of NN.
the learning phase is based on wj = wj + learning_rate * (yi_actual - yi_predicted) * x_ij, b = b + learning_rate * (yi_actual - yi_predicted) 
where j is the feature index

yi_predicted is calculated as:
activation_function(sum(wi * xi) + b) for feature index i 

How the activation function works:
-Takes the weighted sum of inputs: Each neuron receives inputs from the previous layer (or the input layer), 
and each input is multiplied by a corresponding weight. A bias term is also added. This results in a single numerical value.  
-Applies a non-linear transformation: The activation function takes this numerical value as its input 
and applies a mathematical function to it. The key is that most activation functions are non-linear.
-Produces the output: The output of the activation function becomes the output of the neuron, 
which is then passed on to the next layer in the network
'''
class Perceptron:
    #the hyperparameters need to be tuned by k-CV or through sklearn.model_selection.validation_curve()
    def __init__(self, learning_rate = 0.01, n_iterations = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.activationFunc = self._unit_step_func
        self.weights = None
        self.bais = None

    '''
    the learning phase aims to adjust the weights and bias
    '''
    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bais = 0

        y_ = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bais #np.dot(wi * xi) = sum(wi * xi)
                y_predicted = self.activationFunc(linear_output)

                stepUpdate = self.learning_rate * (y_[idx] - y_predicted)
                self.weights += stepUpdate * x_i
                self.bais += stepUpdate
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bais
        y_predicted = self.activationFunc(linear_output)
        return y_predicted

    #binary classification based on positive/negative linear_output
    def _unit_step_func(self, x):
        return np.where(x >= 0, 1, 0)

   
if __name__ == "__main__":
    X, y = datasets.make_blobs(n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=1234)

    print(f'shape {X_train.shape}\n', X_train) #2D dataset

    plt.figure()
    plt.scatter(X[:,0], X[:,1], c=y, edgecolors='k', s=20) #for simplicity of visualiztion, we'll consider the first column as x, and the second column as y 
    #plt.show()

    perc = Perceptron()
    perc.fit(X_train, y_train)
    predictions = perc.predict(X_test)
    accuracy = np.sum(predictions == y_test) / len(y_test)
    print(f'Acc.:{accuracy}') #for classification problems we use the accuracy
    #in case of base of bad accuracy we might try to use preprocessing techniques like "scale standarization"