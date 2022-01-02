import os
import time
import matplotlib.pyplot as plt
from LogisticRegression import logistic_regression
from LRM import logistic_regression_multiclass
from DataReader import *

data_dir = "../data/"
train_filename = "training.npz"
test_filename = "test.npz"

def visualize_features(X, y):
    '''This function is used to plot a 2-D scatter plot of training features.

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.

    Returns:
        No return. Save the plot to 'train_features.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
    plt.style.use('seaborn-whitegrid')
    plt.plot(X[:,0], X[:,1], 'o', markersize=2, color='black')
    plt.title("Visualize feature values - Scatter Plot")
    plt.xlabel("Symmetry feature")
    plt.ylabel("Intensity feature")
    plt.savefig('train_features.png')
    ### END YOUR CODE

def visualize_result(X, y, W):
    '''This function is used to plot the sigmoid model after training.

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.
        W: An array of shape [n_features,].

    Returns:
        No return. Save the plot to 'train_result_sigmoid.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
    intercept = -W[0]/W[2]; slope = -W[1]/W[2]
    plot_x = np.linspace(-1,0,100)
    plot_y = slope*plot_x + intercept
    plt.plot(plot_x, plot_y, '-r')
    plt.style.use('seaborn-whitegrid')
    plt.plot(X[:,0], X[:,1], 'o', markersize=2, color='black')
    plt.title("Visualize Sigmoid Results")
    plt.xlabel("Symmetry feature")
    plt.ylabel("Intensity feature")
    plt.ylim([-1.0, 0])
    plt.xlim([-1.0, 0])
    plt.savefig('train_result_sigmoid.png')
    ### END YOUR CODE

def visualize_result_multi(X, y, W):
    '''This function is used to plot the softmax model after training.

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 0,1,2.
        W: An array of shape [n_features, 3].

    Returns:
        No return. Save the plot to 'train_result_softmax.*' and include it
        in submission.
	'''
	### YOUR CODE HERE
    intercept_1 = -W[0][0]/W[2][0]
    intercept_2 = -W[0][1]/W[2][1]
    intercept_3 = -W[0][2]/W[2][2]

    slope_1 = -W[1][0]/W[2][0]
    slope_2 = -W[1][1]/W[2][1]
    slope_3 = -W[1][2]/W[2][2]

    plot_x = np.linspace(-1,0,100)
    plot_y_1 = slope_1*plot_x + intercept_1
    plot_y_2 = slope_2*plot_x + intercept_2
    plot_y_3 = slope_3*plot_x + intercept_3

    plt.style.use('seaborn-whitegrid')
    plt.plot(plot_x, plot_y_1, '-r')
    plt.plot(plot_x, plot_y_2, '-b')
    plt.plot(plot_x, plot_y_3, '-g')
    plt.plot(X[:,0], X[:,1], 'o', markersize=2, color='black')
    plt.title("Visualize Softmax Results")
    plt.xlabel("Symmetry feature")
    plt.ylabel("Intensity feature")
    plt.ylim([-1.0, 0.5])
    plt.xlim([-1.0, 0])
    plt.savefig('train_result_softmax.png')
	### END YOUR CODE

def main():
	# ------------Data Preprocessing------------
	# Read data for training.

    raw_data, labels = load_data(os.path.join(data_dir, train_filename))
    raw_train, raw_valid, label_train, label_valid = train_valid_split(raw_data, labels, 2300)

    ##### Preprocess raw data to extract features
    train_X_all = prepare_X(raw_train)
    valid_X_all = prepare_X(raw_valid)
    ##### Preprocess labels for all data to 0,1,2 and return the idx for data from '1' and '2' class.
    train_y_all, train_idx = prepare_y(label_train)
    valid_y_all, val_idx = prepare_y(label_valid)

    ####### For binary case, only use data from '1' and '2'
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    ####### Only use the first 1350 data examples for binary training.
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    ####### set lables to  1 and -1. Here convert label '2' to '-1' which means we treat data '1' as postitive class.
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1
    data_shape= train_y.shape[0]

#    # Visualize training data.
    visualize_features(train_X[:, 1:3], train_y)


   # ------------Logistic Regression Sigmoid Case------------

   ##### Check GD, SGD, BGD
    '''
    logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=100)

    #logisticR_classifier.fit_GD(train_X, train_y)
    #print(logisticR_classifier.get_params())
    #print(logisticR_classifier.score(train_X, train_y))

    #logisticR_classifier.fit_BGD(train_X, train_y, data_shape)
    #print(logisticR_classifier.get_params())
    #print(logisticR_classifier.score(train_X, train_y))

    #logisticR_classifier.fit_SGD(train_X, train_y)
    #print(logisticR_classifier.get_params())
    #print(logisticR_classifier.score(train_X, train_y))

    #logisticR_classifier.fit_BGD(train_X, train_y, 1)
    #print(logisticR_classifier.get_params())
    #print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_BGD(train_X, train_y, 10)
    #print(logisticR_classifier.get_params())
    #print(logisticR_classifier.score(train_X, train_y))

    # Explore different hyper-parameters.
    ### YOUR CODE HERE
    learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    max_iters = [100, 200, 300, 400, 500]
    batch_sizes = [1, 5, 10, 15, 20]

    score_GD = 0
    score_BGD = 0
    score_SGD = 0

    hyperparameter_GD = {'learning_rate':0, 'max_iter':0}
    hyperparameter_BGD = {'learning_rate':0, 'max_iter':0, 'batch_size':0}
    hyperparameter_SGD = {'learning_rate':0, 'max_iter':0}

    for rate in learning_rates:
        for iter in max_iters:
            logisticR_classifier = logistic_regression(learning_rate=rate, max_iter=iter)
            logisticR_classifier.fit_GD(train_X, train_y)
            score = logisticR_classifier.score(valid_X, valid_y) # accuarcy score based on validation set
            if(score>score_GD):
                score_GD = score
                hyperparameter_GD['learning_rate'] = rate
                hyperparameter_GD['max_iter'] = iter
            logisticR_classifier.fit_SGD(train_X, train_y)
            score = logisticR_classifier.score(valid_X, valid_y) # accuarcy score based on validation set
            if(score>score_SGD):
                score_SGD = score
                hyperparameter_SGD['learning_rate'] = rate
                hyperparameter_SGD['max_iter'] = iter
            for size in batch_sizes:
                logisticR_classifier.fit_BGD(train_X, train_y, size)
                score = logisticR_classifier.score(valid_X, valid_y) # accuarcy score based on validation set
                if(score>score_BGD):
                    score_BGD = score
                    hyperparameter_BGD['learning_rate'] = rate
                    hyperparameter_BGD['max_iter'] = iter
                    hyperparameter_BGD['batch_size'] = size

    if(score_GD > score_BGD):
        if(score_GD > score_SGD):
            best = 'GD'
            hyperparameters = hyperparameter_GD
        else:
            best = 'SGD'
            hyperparameters = hyperparameter_SGD
    elif(score_BGD > score_SGD):
        best = 'BGD'
        hyperparameters = hyperparameter_BGD
    else:
        best = 'SGD'
        hyperparameters = hyperparameter_SGD

    total_training_X = np.concatenate((train_X, valid_X))
    total_training_y = np.concatenate((train_y, valid_y))

    best_logisticR_classifier = logistic_regression(learning_rate=hyperparameters['learning_rate'], max_iter=hyperparameters['max_iter'])
    if(best == 'GD'):
        best_logisticR_classifier.fit_GD(total_training_X, total_training_y)
    elif(best == 'BGD'):
        best_logisticR_classifier.fit_BGD(total_training_X, total_training_y, hyperparameters['batch_size'])
    else:
        best_logisticR_classifier.fit_SGD(total_training_X, total_training_y)
    ### END YOUR CODE

	# Visualize the your 'best' model after training.

    ### YOUR CODE HERE
    visualize_result(train_X[:, 1:3], train_y, best_logisticR_classifier.get_params())
    print('Binary Logistic Regression hyperparameters')
    print(hyperparameters)
    ### END YOUR CODE

    # Use the 'best' model above to do testing. Note that the test data should be loaded and processed in the same way as the training data.
    ### YOUR CODE HERE
    raw_test_data, test_labels = load_data(os.path.join(data_dir, test_filename))
    test_X = prepare_X(raw_test_data)
    test_y, test_idx = prepare_y(test_labels)
    test_X = test_X[test_idx]
    test_y = test_y[test_idx]
    test_y[np.where(test_y==2)] = -1

    test_score = best_logisticR_classifier.score(test_X, test_y)
    print('Accuracy Score for Binary Classification Test Set',test_score)
    ### END YOUR CODE
    '''
    # ------------Logistic Regression Multiple-class case, let k= 3------------
    ###### Use all data from '0' '1' '2' for training
    train_X = train_X_all
    train_y = train_y_all
    valid_X = valid_X_all
    valid_y = valid_y_all

    #########  BGD for multiclass Logistic Regression
    #logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=100,  k=3)
    #logisticR_classifier_multiclass.fit_BGD(train_X, train_y, 10)
    #print(logisticR_classifier_multiclass.get_params())
    #print(logisticR_classifier_multiclass.score(train_X, train_y))

    # Explore different hyper-parameters.
    ### YOUR CODE HERE

    learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    max_iters = [100, 200, 300, 400, 500]
    batch_sizes = [1, 5, 10, 15, 20]

    final_score = 0
    hyperparameters = {'learning_rate':0, 'max_iter':0, 'batch_size':0}
    for rate in learning_rates:
        for iter in max_iters:
            logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=rate, max_iter=iter,  k=3)
            for size in batch_sizes:
                logisticR_classifier_multiclass.fit_BGD(train_X, train_y, size)
                score = logisticR_classifier_multiclass.score(valid_X, valid_y) # accuarcy score based on validation set
                if(score > final_score):
                    final_score = score
                    hyperparameters['learning_rate'] = rate
                    hyperparameters['max_iter'] = iter
                    hyperparameters['batch_size'] = size

    total_training_X = np.concatenate((train_X, valid_X))
    total_training_y = np.concatenate((train_y, valid_y))

    best_logistic_multi_R_classifier = logistic_regression_multiclass(learning_rate=hyperparameters['learning_rate'], max_iter=hyperparameters['max_iter'],  k=3)
    best_logistic_multi_R_classifier.fit_BGD(total_training_X, total_training_y, hyperparameters['batch_size'])
    ### END YOUR CODE

    # Visualize the your 'best' model after training.
    visualize_result_multi(train_X[:, 1:3], train_y, best_logistic_multi_R_classifier.get_params())
    print('Multiple Logistic Regression hyperparameters')
    print(hyperparameters)

    # Use the 'best' model above to do testing.
    ### YOUR CODE HERE
    raw_test_data, test_labels = load_data(os.path.join(data_dir, test_filename))
    test_X = prepare_X(raw_test_data)
    test_y, test_idx = prepare_y(test_labels)

    test_score = best_logistic_multi_R_classifier.score(test_X, test_y)
    print('Accuracy Score for Multi Class Classification Test Set',test_score)
    ### END YOUR CODE

    # ------------Connection between sigmoid and softmax------------
    ############ Now set k=2, only use data from '1' and '2'
    '''
    #####  set labels to 0,1 for softmax classifer
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    train_y[np.where(train_y==2)] = 0
    valid_y[np.where(valid_y==2)] = 0

    ###### First, fit softmax classifer until convergence, and evaluate
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE
    time_1 = time.time() # for calculating time required for training
    learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    batch_sizes = [1, 5, 10, 15, 20]

    final_score = 0
    hyperparameters_MLR = {'learning_rate':0, 'batch_size':0}
    for rate in learning_rates:
        logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=rate, max_iter=10000,  k=2)
        for size in batch_sizes:
            logisticR_classifier_multiclass.fit_BGD(train_X, train_y, size)
            score = logisticR_classifier_multiclass.score(valid_X, valid_y) # accuarcy score based on validation set
            if(score > final_score):
                final_score = score
                hyperparameters_MLR['learning_rate'] = rate
                hyperparameters_MLR['batch_size'] = size

    total_training_X_MLR = np.concatenate((train_X, valid_X))
    total_training_y_MLR = np.concatenate((train_y, valid_y))

    best_logistic_multi_R_classifier = logistic_regression_multiclass(learning_rate=hyperparameters_MLR['learning_rate'], max_iter=10000,  k=2)
    best_logistic_multi_R_classifier.fit_BGD(total_training_X_MLR, total_training_y_MLR, hyperparameters_MLR['batch_size'])

    raw_test_data, test_labels = load_data(os.path.join(data_dir, test_filename))
    test_X = prepare_X(raw_test_data)
    test_y, test_idx = prepare_y(test_labels)
    test_X = test_X[test_idx]
    test_y = test_y[test_idx]
    test_y[np.where(test_y==2)] = 0

    print('Accuracy for Test Set by Multiclass Logistic Regression when trained over same set',best_logistic_multi_R_classifier.score(test_X, test_y))
    print(time.time() - time_1) # for calculating time required for training
    ### END YOUR CODE



    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    #####       set lables to -1 and 1 for sigmoid classifer
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1

    ###### Next, fit sigmoid classifer until convergence, and evaluate
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE
    time_1 = time.time() # for calculating time required for training

    learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    batch_sizes = [1, 5, 10, 15, 20]

    final_score = 0
    hyperparameters_LR = {'learning_rate':0, 'batch_size':0}
    for rate in learning_rates:
        logisticR_classifier = logistic_regression(learning_rate=rate, max_iter=10000)
        for size in batch_sizes:
            logisticR_classifier.fit_BGD(train_X, train_y, size)
            score = logisticR_classifier.score(valid_X, valid_y) # accuarcy score based on validation set
            if(score > final_score):
                final_score = score
                hyperparameters_LR['learning_rate'] = rate
                hyperparameters_LR['batch_size'] = size

    total_training_X_LR = np.concatenate((train_X, valid_X))
    total_training_y_LR = np.concatenate((train_y, valid_y))

    best_logistic_classifier = logistic_regression(learning_rate=hyperparameters_LR['learning_rate'], max_iter=10000)
    best_logistic_classifier.fit_BGD(total_training_X_LR, total_training_y_LR, hyperparameters_LR['batch_size'])

    raw_test_data, test_labels = load_data(os.path.join(data_dir, test_filename))
    test_X = prepare_X(raw_test_data)
    test_y, test_idx = prepare_y(test_labels)
    test_X = test_X[test_idx]
    test_y = test_y[test_idx]
    test_y[np.where(test_y==2)] = -1

    print('Accuracy for Test Set by Binary Logistic Regression when trained over same set',best_logistic_classifier.score(test_X, test_y))
    print(time.time() - time_1) # for calculating time required for training
    '''
    ### END YOUR CODE
    ################Compare and report the observations/prediction accuracy

    '''
    Explore the training of these two classifiers and monitor the graidents/weights for each step.
    Hint: First, set two learning rates the same, check the graidents/weights for the first batch in the first epoch. What are the relationships between these two models?
    Then, for what leaning rates, we can obtain w_1-w_2= w for all training steps so that these two models are equivalent for each training step.

    ### YOUR CODE HERE
    best_logistic_multi_R_classifier = logistic_regression_multiclass(learning_rate=hyperparameters_MLR['learning_rate'], max_iter=10000,  k=2)
    best_logistic_multi_R_classifier.fit_BGD(total_training_X_MLR, total_training_y_MLR, hyperparameters_MLR['batch_size'])
    print(best_logistic_multi_R_classifier.get_params())

    best_logistic_classifier = logistic_regression(learning_rate=hyperparameters_LR['learning_rate'], max_iter=10000)
    best_logistic_classifier.fit_BGD(total_training_X_LR, total_training_y_LR, hyperparameters_LR['batch_size'])
    print(best_logistic_classifier.get_params())
    ### END YOUR CODE
    '''
    # ------------End------------


if __name__ == '__main__':
	main()
