import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

'''
Unlike Principal Component Analysis (PCA), which is an unsupervised dimensionality reduction technique 
that focuses on maximizing the variance of the data (regardless of class labels),
LDA is a supervised technique. Its core goal is to find a linear combination of features that maximizes 
the separation between classes while minimizing the variance within each class.
[Aims to reduce the dimensionality] + Can be use as a preprocessing step to improve the performance of other classifiers,
since it builds decision boundaries by maximize class separability.

Steps:
1-Calculate the mean vectors for each class.
2-Compute the within-class scatter matrix (SW) and the between-class scatter matrix (SB).
3-Solve the generalized eigenvalue problem for SW^-1 * SB.
4-Select the eigenvectors (linear discriminants) corresponding to the largest eigenvalues. The maximum number of discriminants is limited by the number of classes minus one (C−1) or the number of features, whichever is smaller.
5-Project the original data onto the new lower-dimensional subspace using these selected eigenvectors.
'''
class LDA:
    '''
    n_components allows to control the trade-off between dimensionality reduction and information retention.
    n_components [0:1] preserves at least the determined values of data variance.
    n_components [1 <= n] preserves the exact number of features requested
    '''
    def __init__(self, n_components = 0.90):
        self.n_components = n_components
        self.linear_discriminants = None
        self.mean = None
        self.explained_variance = None # Variance explained by each component
        self.explained_discriminant_ratio = None # Ratio of discriminant info per component

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        class_labels = np.unique(y)
        
        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))

        self.mean = np.mean(X, axis=0)
        
        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)

            S_W += (X_c - mean_c).T.dot((X_c - mean_c))
            n_c = X_c.shape[0]
            mean_diff = (mean_c - self.mean).reshape(n_features, 1)

            S_B += n_c * (mean_diff).dot(mean_diff.T)

        # # Handle potential singularity of S_W before inversion
        # # Add a small regularization term to the diagonal of S_W
        # epsilon = 1e-6
        # S_W_regularized = S_W + epsilon * np.eye(n_features)
        
        # # Check if S_W_regularized is still singular (rare but possible with pathological data)
        # try:
        #     S_W_inv = np.linalg.inv(S_W_regularized)
        # except np.linalg.LinAlgError:
        #     print("Warning: S_W is singular even after regularization. This might lead to unstable results.")
        #     # Fallback strategy: PCA on S_W or further regularization
        #     S_W_inv = np.linalg.pinv(S_W_regularized) # Use pseudo-inverse as a fallback
            
        inversed_matrix = np.linalg.inv(S_W).dot(S_B)

        eigen_values, eigen_vectors = np.linalg.eig(inversed_matrix)
        idxs = np.argsort(eigen_values)[::-1]
        eigen_values = eigen_values[idxs] #each value represents a feature
        eigen_vectors = eigen_vectors[:, idxs] #each vector represents an instance of X

        # Determine the number of components to keep
        if self.n_components is None:
            self.n_components = n_features
        elif isinstance(self.n_components, int):
            self.n_components = min(self.n_components, n_features)
        elif 0 < self.n_components < 1:
            cumulative_variance_ratio = np.cumsum(eigen_values) / np.sum(eigen_values)
            self.n_components = np.argmax(cumulative_variance_ratio >= self.n_components) + 1
        else:
            raise ValueError("Invalid value for n_components.")
        
        self.linear_discriminants = eigen_vectors[:, :self.n_components]
        self.explained_variance = eigen_values[:self.n_components]
        self.explained_variance_ratio = self.explained_variance / np.sum(eigen_values)

        print(f"explained variance per comp. idxs{idxs[:self.n_components]}: {self.explained_variance}")
        print(f"explained variance ratio per comp. idxs{idxs[:self.n_components]}: {self.explained_variance_ratio}")

    
    def transform(self, X):
        if self.linear_discriminants is None:
            raise RuntimeError("LDA has not been fitted. Call fit() first.")
        
        transformed_X = np.dot(X, self.linear_discriminants)
        return transformed_X

    #approximated reconstruction
    def inverse_transform(self, X_transformed):
        if self.linear_discriminants is None:
            raise RuntimeError("LDA has not been fitted. Call fit() first.")
            
        reconstructed_X = np.dot(X_transformed, self.linear_discriminants.T)
        return reconstructed_X

if __name__ == "__main__":
    X, y = datasets.make_blobs(n_samples=150, n_features=5, centers=2, cluster_std=1.05, random_state=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=1234)

    print(f'shape {X_train.shape}\n', X_train) #5D dataset

    # plt.figure()
    # plt.scatter(X[:,0], X[:,1], c=y, edgecolors='k', s=20) #for simplicity of visualiztion, we'll consider the first column as x, and the second column as y 
    # plt.show()

    lda = LDA(3)
    lda.fit(X_train, y_train)

    transformed_X = lda.transform(X_train)
    print(f'transformed shape {transformed_X.shape}\n', transformed_X)

    reconstructed_X = lda.inverse_transform(transformed_X)
    print(f'reconstructed shape {reconstructed_X.shape}\n', reconstructed_X)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(X_train[:,0], X_train[:,1], c=y_train, edgecolors='k', s=20) #for simplicity of visualiztion, we'll consider the first column as x, and the second column as y 
    plt.title('Original X')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.scatter(reconstructed_X[:,0], reconstructed_X[:,1], c=y_train, edgecolors='k', s=20) #for simplicity of visualiztion, we'll consider the first column as x, and the second column as y 
    plt.title('LDA Reconstruction')
    plt.legend()
    plt.grid(True)
    plt.show()
