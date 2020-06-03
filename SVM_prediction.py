# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:36:57 2020

@author: The Jipsess
"""
from sklearn import svm, metrics
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import numpy as np
from numpy import random

def fit(X_train, Y_train, X_test, Y_test, num_features = 2):
    
    if num_features == 2:
        clf = svm.SVC()
        SVM_model = clf.fit(X_train, Y_train)
    elif num_features > 2:
        print('multi-class classification is not yet implemented for SVM')
    else:
        raise Exception('Classifying single featured data is redundant!') 
    
    SVM_score = SVM_model.score(X_test, Y_test)
    
    # print final results
    print(F"Support Vector Machine accuracy: {SVM_score}\n")
    
    return (SVM_model)
    
# %% Find the optimal paramters
    
def hyperparameter_tuning(X_train, Y_train, X_test, Y_test):
    classifier =  svm.SVC(cache_size = 1024,
                          class_weight = 'balanced',
                          random_state = random.randint(1,10000))
    
    # Set possible parameter values
    C = [0.01, 0.1, 1, 10]
    gamma = [0.001, 0.01, 0.1]
    kernel = ['linear', 'rbf', 'poly', 'sigmoid']
    tol = [0.0001, 0.001, 0.01]
    
    # Wrap the parameter space in random_grid
    random_grid = {'C' : C,
                   'gamma' : gamma,
                   'kernel' : kernel,
                   'tol' : tol}
    
    #===Run a randomised search for the optimal parameter setting===#
    classifier_random = RandomizedSearchCV(estimator = classifier,
                                       param_distributions = random_grid,
                                       n_iter = 100,
                                       cv = 5,
                                       verbose = 2,
                                       random_state = random.randint(1,10000),
                                       n_jobs = -1)
    
    # Fit the random search model
    param_opt_rand = classifier_random.fit(X_train, Y_train)
    
    #=== Narrow down random optimal solutions to the best hyperparamets ===#
    # Extrapolate random optimal parameter values
    tol = param_opt_rand.best_estimator_.tol
    kernel = param_opt_rand.best_estimator_.kernel
    gamma = param_opt_rand.best_estimator_.gamma
    C = param_opt_rand.best_estimator_.C
    
    
    # IDEA: implement a precision variable. Divide the step size and the offset
    # used in the np.arange() by the precision which should by default be 1.
    param_grid = {'C' : np.arange(C - 0.8, C + 0.81, 0.4),
                  'gamma' : [0.5 * gamma,
                             gamma,
                             gamma * 2],
                  'tol' : [0.5 * tol,
                           tol,
                           2 * tol]}
    # Set kernel
    classifier.kernel = kernel
    
    # Find the optimal hyperparameters using gridsearch
    # initialise the grid search with cross validation
    classifier_gridsearch = GridSearchCV(estimator = classifier,
                               param_grid = param_grid, 
                               cv = 5, n_jobs = -1, verbose = 2)
    # Run the grid search to find the model with the optimal hyperparamterers
    classifier_gridsearch.fit(X_train, Y_train)
    
    # Extrapolate the optimal hyperparamters
    SVC_params_opt = classifier_gridsearch.best_params_
    
    # Evaluate the optimal model
    final_accuracy = classifier_gridsearch.best_estimator_.score(X_test,
                                                                 Y_test)
    final_roc_auc_score = metrics.roc_auc_score(Y_test,
                                                classifier_gridsearch.best_estimator_.predict(X_test))

    # print the results
    print("Support Vector Machine hyperparamter optimisation completed succesfully.")
    print(f"Best SVM model accuracy: {final_accuracy}")
    print(f"Best SVM model roc_auc: {final_roc_auc_score}")
    
    return classifier_gridsearch.best_estimator_
