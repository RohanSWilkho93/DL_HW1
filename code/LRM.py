#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 12:00:48 2019

@author:
"""
import numpy as np
import random
import math
import sys

"""This script implements a two-class logistic regression model.
"""

class logistic_regression_multiclass(object):

    def __init__(self, learning_rate, max_iter, k):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.k = k

    def fit_BGD(self, X, labels, batch_size):
        """Train perceptron model on data (X,y) with BGD.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,].  Only contains 0,..,k-1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.

        Hint: the labels should be converted to one-hot vectors, for example: 1----> [0,1,0]; 2---->[0,0,1].
        """

		### YOUR CODE HERE
        #random.seed(42)
        n_samples, n_features = X.shape

        # convertng labels to one-hot vector
        one_hot_encoding = np.zeros((labels.size, self.k))
        labels = labels.astype(int)
        one_hot_encoding[np.arange(labels.size), labels] = 1

        self.W = np.zeros((n_features, self.k))
        for i in range(self.max_iter):
            prev_iter = 0 # the previous iteration where the batch update took place
            for j in range(n_samples):
                if(j - prev_iter == batch_size): # checking if the batch size is reached or not
                    random_number = random.randint(prev_iter, j) # selecting a random number in the batch
                    prev_iter = j # updating the previous iteration
                    gradient = self._gradient(X[random_number], one_hot_encoding[random_number]) # calculating the gradient
                    #if(np.linalg.norm(gradient*1./batch_size) < 0.0005): break
                    self.W = (np.subtract(self.W,self.learning_rate*gradient)) # updating the weights

            #if(np.linalg.norm(gradient*1./batch_size) < 0.0005): break

            if(not(X.shape[0]%batch_size == 0)): # if the sample size is not completely divisible by batch_size, certain samples will be left out of training

                random_number = random.randint(prev_iter, j-1) # selecting a random number in the remaining samples
                prev_iter = j # updating the previous iteration
                gradient = self._gradient(X[random_number], one_hot_encoding[random_number]) # calculating the gradient
                #if(np.linalg.norm(gradient*1./batch_size) < 0.0005): break
                self.W = (np.subtract(self.W,self.learning_rate*gradient)) # updating the weights




		### END YOUR CODE


    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: One_hot vector.

        Returns:
            _g: An array of shape [n_features,]. The gradient of
                cross-entropy with respect to self.W.
        """
		### YOUR CODE HERE
        weigths_transpose = self.W.transpose()
        product = np.dot(weigths_transpose, _x)
        soft_max = self.softmax(product)
        _g = (_x*(soft_max - _y).reshape(self.k,1)).transpose()
        return(_g)
		### END YOUR CODE

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        ### You must implement softmax by youself, otherwise you will not get credits for this part.

		### YOUR CODE HERE
        soft_max = np.zeros(x.shape)
        for i in range(soft_max.shape[0]):
        	soft_max[i] = math.exp(x[i])
        soft_max = soft_max/np.sum(soft_max)
        return(soft_max)
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


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 0,..,k-1.
        """
		### YOUR CODE HERE
        n_samples, n_features = X.shape
        pred = list()
        for i in range(X.shape[0]):
            product = list(np.dot(self.W.transpose(), X[i]))
            max_value = max(product)
            pred.append(product.index(max_value))
        return(np.asarray(pred))
		### END YOUR CODE


    def score(self, X, labels):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,]. Only contains 0,..,k-1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. labels.
        """
		### YOUR CODE HERE
        predictions = self.predict(X)

        num_correct_pred = np.sum(predictions == labels)
        score = num_correct_pred/predictions.shape[0]

        return(score)
		### END YOUR CODE
