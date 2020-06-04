# -*- coding: utf-8 -*-
"""
Created on Tue May  5 15:07:56 2020

@author: The Jipsess

WARNING: Are you not using CUDA? Or have you never even heard of it? Then it
might take very long to train the ANN. In this case please refer to the end of
this script where grid_search is defined and set n_jobs=-1. 

"""


# %% Load all the required packages
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2

# %% The function to optimise the hyperparamters and return the optimal model
def hyperparameter_tuning(X_train, Y_train, X_test, Y_test,
                          balance_class_weights = False):

    # %% Build & fit a single Artificial Neural Network.
    def build_single_ANN():
        # Initialising the classifier
        classifier = Sequential()
        
        # Add the input layer and the first hidden layer with Dropout
        classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu', input_dim = 12328))
        classifier.add(Dropout(rate = 0.1))
        
        # Adding the second hidden layer with Dropout
        classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))
        classifier.add(Dropout(rate = 0.1))
        
        # Adding the second hidden layer with Dropout
        classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))
        classifier.add(Dropout(rate = 0.1))
        
        # Adding the second hidden layer with Dropout
        classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))
        # Dropout is a form of regularisation to prevent overfitting
        classifier.add(Dropout(rate = 0.1))
        
        # Adding the output layer, we have an output array of one column, thus units = 1
        classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
        
        # Compiling the ANN
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
        
        # Fitting the ANN to the Training set
        classifier.fit(X_train, Y_train, batch_size = 100, epochs = 100)
        
        # evaluate model.
        # Predicting the Test set results
        Y_pred = classifier.predict(X_test)
        Y_pred[Y_pred > 0.5] = 1
        Y_pred[Y_pred != 1] = 0
        
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(Y_test, Y_pred)
        
        def accuracy(confusion_matrix):
            diagonal_sum = confusion_matrix.trace()
            sum_of_all_elements = confusion_matrix.sum()
            return diagonal_sum / sum_of_all_elements 
        
        accuracy(cm)
    
    # %% Model Parameter optimisation
    
    
    
    def build_model(tuned_optimiser,
                    architecture = 1,
                    regularisation_amount = 0.1,
                    hidden_layers = 3):
        
        if architecture == 1:
            # Initialising the classifier
            classifier = Sequential()
            
            # Add the input layer and the first hidden layer with Dropout
            classifier.add(Dense(units = 64,
                                 kernel_initializer = 'uniform',
                                 activation = 'relu',
                                 input_dim = X_train.shape[1]))
            classifier.add(Dropout(rate = regularisation_amount))
            
            # Adding the hidden layers
            for i in range(hidden_layers):
                # Add dense layer
                classifier.add(Dense(units = 64,
                                     kernel_initializer = 'uniform',
                                     activation = 'relu'))
                # Add droput layer for regularisation
                classifier.add(Dropout(rate = regularisation_amount))
            
            # Add the output layer, we have an output array of one column, thus units = 1
            classifier.add(Dense(units = 1, kernel_initializer = 'uniform',
                                 activation = 'sigmoid'))
            
            # Compiling the ANN
            classifier.compile(optimizer = tuned_optimiser, loss = 'binary_crossentropy',
                               metrics = ['binary_accuracy'])
        
        elif architecture == 2:
            # Initialising the sequential model
            classifier = Sequential()
            
            # Add the input layer and the first hidden layer with L2 regularisation
            classifier.add(Dense(units = 64,
                                 kernel_initializer = 'uniform',
                                 activation = 'relu',
                                 kernel_regularizer = l2(regularisation_amount),
                                 input_dim = X_train.shape[1]))
            
            # Adding the hidden layers
            for i in range(hidden_layers):
                classifier.add(Dense(units = 64,
                                     kernel_initializer = 'uniform',
                                     kernel_regularizer = l2(regularisation_amount),
                                     activation = 'relu'))
            
            # Add the output layer
            classifier.add(Dense(units = 1,
                                 kernel_initializer = 'uniform',
                                 activation = 'sigmoid'))
            
            # Compile the ANN
            classifier.compile(optimizer = tuned_optimiser,
                               loss = 'binary_crossentropy',
                               metrics = ['binary_accuracy'])
        elif architecture == 3:
            # Initialising the sequential model
            classifier = Sequential()
            
            # Add the input layer and the first hidden layer with L2 regularisation
            classifier.add(Dense(units = 30,
                                 kernel_initializer = 'uniform',
                                 activation = 'relu',
                                 kernel_regularizer = l2(regularisation_amount),
                                 input_dim = X_train.shape[1]))
            
            # Add the output layer
            classifier.add(Dense(units = 1,
                                 kernel_initializer = 'uniform',
                                 activation = 'sigmoid'))
            
            # Compiling the ANN
            classifier.compile(optimizer = tuned_optimiser, loss = 'binary_crossentropy',
                               metrics = ['accuracy'])
            
        else:
            raise Exception('Unknown Architecture')
            
        # Return the model
        return classifier
    
    # Balance the class weights if requested
    if balance_class_weights == True:
        # Set class weight to account for imbalanced data
        one_count = np.sum(Y_train)
        zero_count = Y_train.shape[0] - one_count
        one_weight = one_count / Y_train.shape[0]
        zero_weight = zero_count / Y_train.shape[0]
        classweight = {0 : zero_weight,
                       1 : one_weight}
    else:
        classweight = {0 : 1,
                       1 : 1}
    
    # Intialise the classifier
    classifier = KerasClassifier(build_fn = build_model,
                                 batch_size = 10,
                                 epochs = 100)    
    
    # Define hyperparamater space (all possible combinations will be tested)
    # If the it takes to long to run, reduce the number of values
    parameters = {'batch_size': [64, 32, 12],
                  'epochs': [20, 100],
                  'tuned_optimiser': ['adam',  'SGD'],
                  'regularisation_amount' : [0.1, 0.4, 0.5],
                  'architecture' : [1],
                  'hidden_layers' : [2]}
    
    # Intialisethe cross validation grid search
    # If you're not using CUDA or don't know what that is, try n_jobs=-1
    grid_search = GridSearchCV(estimator = classifier,
                               param_grid = parameters,
                               scoring = 'roc_auc',
                               cv = 5,
                               n_jobs = 1,
                               verbose = 2,
                               return_train_score = True,
                               iid = True)
    
    # Run the grid search to find the best model
    grid_search = grid_search.fit(X_train,Y_train, class_weight = classweight)
    
    # Extrapolate some information from the gridsearch results (redundant)
    best_parameters = grid_search.best_params_
    best_accuracy = grid_search.best_score_
    best_model = grid_search.best_estimator_.model
    
    # Predict the Test set results
    y_pred = grid_search.best_estimator_.predict(X_test)
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(Y_test, y_pred)
    
    # Create function to calculate the accuracy
    def accuracy(confusion_matrix):
        diagonal_sum = confusion_matrix.trace()
        sum_of_all_elements = confusion_matrix.sum()
        return diagonal_sum / sum_of_all_elements 
    
    # Calculate the accuracy on the test set
    acc = accuracy(cm)
    print(f'Finnished! The best ANN has an accuracy of {acc} on the test set')
    
    return(grid_search.best_estimator_)
