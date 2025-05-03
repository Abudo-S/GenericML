import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

'''
One level simple decision tree predictor for a single feature.
Polarity dictates which side corresponds to which class label (-1 or +1).
Threshold is a single unique value of a certain feature.
Alpha is the assigned weight to a decision stump
'''
class DecisionStump:
    alpha : float

    def __init__(self):
        self.polarity = 1
        self.featureIdx = None
        self.alpha = None
        self.threshold = None

    def predict(self, X: np.ndarray):
        n_samples = X.shape[0]
        X_feature_values = X[:, self.featureIdx]

        predictions = np.ones(n_samples) #initial predictions

        #for i, element-wise comparison between each value in the X_column array and the self.threshold 
        if self.polarity == 1:
            predictions[X_feature_values < self.threshold] = -1
        else: #polarity = -1
            predictions[X_feature_values > self.threshold] = -1
        
        return predictions

'''
special cases when weighted_error in {0, 0.5, 1} => {Overly influential, No influence, Inverted prediction}.
'''
class AdaBoost:
    def __init__(self, n_classifiers = 5):
        self.n_classifiers = n_classifiers
        self.classifiers = [DecisionStump]
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape

        weights = np.full(n_samples, (1/n_samples)) #weight initialization
        self.classifiers = []

        for _ in range(self.n_classifiers): #boosting round
            classifier = DecisionStump()

            min_weighted_error = float('inf') #positive infinity

            for featureIdx in range(n_features):
                X_feature_values = X[:, featureIdx]
                thresholds = np.unique(X_feature_values)

                for threshold in thresholds:
                    polarity = 1
                    predictions = np.ones(n_samples)
                    predictions[X_feature_values < threshold] = -1 

                    misclassified_weights = weights[y != predictions]
                    weighted_error = sum(misclassified_weights)

                    if weighted_error > 0.5:
                        weighted_error = 1 - weighted_error
                        polarity = -1

                    if weighted_error < min_weighted_error:
                        min_weighted_error = weighted_error
                        classifier.polarity = polarity
                        classifier.threshold = threshold
                        classifier.featureIdx = featureIdx
            
            ESP = 1e-10
            #update classifier assigned weight
            classifier.alpha = 0.5 * np.log((1 - min_weighted_error + ESP)/(min_weighted_error + ESP))
            predictions = classifier.predict(X)

            #update weights w.r.t. the current classifier
            weights *= np.exp(-classifier.alpha * y * predictions)
            weights /= np.sum(weights)

            self.classifiers.append(classifier)

    #binary {-1, 1} prediction
    #F(x)= t=1∑T ​αt.​ht​(x)
    def predict(self, X: np.ndarray):
        y_pred = np.sum(np.array([classifier.alpha * classifier.predict(X) for classifier in self.classifiers]))

        return np.sign(y_pred)
    

if __name__ == "__main__":
    iris_dataset = datasets.load_breast_cancer()
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

    adaboost = AdaBoost(3)
    adaboost.fit(X_train, y_train)
    predictions = adaboost.predict(X_test)
    accuracy = np.sum(predictions == y_test) / len(y_test)
    print(f'Acc.:{accuracy}') #for classification problems we use the accuracy