import numpy as np
import math
import random
import sys

"""This script implements a two-class logistic regression model.
"""

class logistic_regression(object):

    def __init__(self, learning_rate, max_iter):
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit_GD(self, X, y):
        """Train perceptron model on data (X,y) with GD.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
        n_samples, n_features = X.shape

        ### YOUR CODE HERE
        #self.W = np.zeros((n_features, 1)) # initializing weights
        self.W = [0 for i in range(n_features)] # initializing weights

        for i in range(self.max_iter):
            for j in range(n_samples):
                gradient = self._gradient(X[j], y[j]) # calculating the gradient
                prod = [self.learning_rate*grad for grad in gradient]
                self.W = [self.W[k] - prod[k] for k in range(len(self.W))] # updating the weights
		### END YOUR CODE
        return self

    def fit_BGD(self, X, y, batch_size):
        """Train perceptron model on data (X,y) with BGD.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.
        """
		### YOUR CODE HERE
        random.seed(42)
        n_samples, n_features = X.shape
        self.W = [0 for i in range(n_features)] # initializing weights

        for i in range(self.max_iter):
            prev_iter = 0 # the previous iteration where the batch update took place
            for j in range(n_samples):

                if(j - prev_iter == batch_size): # checking if the batch size is reached or not

                    random_number = random.randint(prev_iter, j) # selecting a random number in the batch
                    prev_iter = j # updating the previous iteration
                    gradient = self._gradient(X[random_number], y[random_number]) # calculating the gradient
                    if(np.linalg.norm([grad/batch_size for grad in gradient]) < 0.0005): break
                    prod = [self.learning_rate*grad for grad in gradient]
                    self.W = [self.W[k] - prod[k] for k in range(len(self.W))] # updating the weights

            if(np.linalg.norm([grad/batch_size for grad in gradient]) < 0.0005): break
            if(not(X.shape[0]%batch_size == 0)): # if the sample size is not completely divisible by batch_size, certain samples will be left out of training

                random_number = random.randint(prev_iter, j-1) # selecting a random number in the remaining samples
                prev_iter = j # updating the previous iteration
                gradient = self._gradient(X[random_number], y[random_number]) # calculating the gradient
                if(np.linalg.norm([grad/batch_size for grad in gradient]) < 0.0005): break
                prod = [self.learning_rate*grad for grad in gradient]
                self.W = [self.W[k] - prod[k] for k in range(len(self.W))] # updating the weights

		### END YOUR CODE
        return self

    def fit_SGD(self, X, y):
        """Train perceptron model on data (X,y) with SGD.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
		### YOUR CODE HERE
        random.seed(42)
        n_samples, n_features = X.shape
        self.W = [0 for i in range(n_features)] # initializing weights
        for i in range(self.max_iter):
            random_number = random.randint(0, n_samples-1) # selecting a random number in the entire sample size
            gradient = self._gradient(X[random_number], y[random_number]) # calculating the gradient
            prod = [self.learning_rate*grad for grad in gradient]
            self.W = [self.W[k] - prod[k] for k in range(len(self.W))] # updating the weights
		### END YOUR CODE
        return self

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: An integer. 1 or -1.

        Returns:
            _g: An array of shape [n_features,]. The gradient of
                cross-entropy with respect to self.W.
        """
		### YOUR CODE HERE
        weigths_transpose = np.asarray(self.W).transpose()
        product = np.dot(weigths_transpose, _x)

        part_one = -_y/(1+math.exp(_y*product))
        _g = list(part_one*_x)

        return(_g)
		### END YOUR CODE

    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features,].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W

    def predict_proba(self, X):
        """Predict class probabilities for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds_proba: An array of shape [n_samples, 2].
                Only contains floats between [0,1].
        """
		### YOUR CODE HERE
        preds_proba = [[0 for i in range(2)] for i in range(X.shape[0])]

        for i in range(len(preds_proba)):
            z = np.dot(np.asarray(self.W).transpose(), X[i])
            sig_z = 1/(1+math.exp(-z))
            preds_proba[i][0] = sig_z
            preds_proba[i][1] = 1 - sig_z

        preds_proba = np.asarray(preds_proba)
        return(preds_proba)
		### END YOUR CODE


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 1 or -1.
        """
		### YOUR CODE HERE
        probabilities = self.predict_proba(X)[:,[0]]

        preds = list()
        for i in range(len(probabilities)):
            if(probabilities[i][0] > 0.5): preds.append(1)
            else: preds.append(-1)

        preds = np.asarray(preds)
        return(preds)
		### END YOUR CODE

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. y.
        """
		### YOUR CODE HERE
        predictions = self.predict(X)

        num_correct_pred = np.sum(predictions == y)
        score = num_correct_pred/predictions.shape[0]

        return(score)
		### END YOUR CODE

    def assign_weights(self, weights):
        self.W = weights
        return self
