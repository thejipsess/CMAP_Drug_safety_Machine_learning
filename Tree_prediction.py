# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# %% Import packages
from sklearn import tree, feature_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, RepeatedStratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
import numpy as np
from numpy import random
import pickle

# %% Fit the model
def fit(X_train, Y_train, X_test, Y_test,
        use_local_parameters = False):
    
    # %% 1 build a decision tree model
    
    # train the decision tree model
    classifier = tree.DecisionTreeClassifier()
    tree_model = classifier.fit(X_train, Y_train)
    
    # Evaluate the performance of the model
    tree_score = tree_model.score(X_test,Y_test)
    
    # %% 2 Build a random forests algorithm
    if use_local_parameters:
        try:
            # Load the local file containing the hyperparameter settings
            directory = "Hyperparameters/RandomForests.pkl"
            RandomForests_params_file = open(directory, "rb")
            params = pickle.load(RandomForests_params_file)
            RandomForests_params_file.close()
            
            # Intiliase random forests classifier with local hyperparameters
            classifier = RandomForestClassifier(
                bootstrap = params['bootstrap'],
                max_features = params['max_features'],
                min_samples_leaf = params['min_samples_leaf'],
                min_samples_split = params['min_samples_split'],
                n_estimators = params['n_estimators'])
            
        except Exception as e:
            print('Could not load local hyperparameters!')
            print('Error: ' + str(e))
            print('Will now continue with default paramters.')
            # Intiliase random forests classifier with deafault hyperparameters
            classifier = RandomForestClassifier()
    else:
        # Intiliase random forests classifier with deafault hyperparameters
        classifier = RandomForestClassifier()
    
    # Assign a random seed    
    classifier.random_state  = random.randint(1,10000)
    # Tell the classifier to use all CPU cores when fitting
    classifier.n_jobs = -1
    # Tell the classifier to 
    classifier.warm_start = False
    # Tell it to balance the class weights
    classifier.class_weight = 'balanced'
    
    # Train the Random Forests model on top of the previous built model
    forest_model = classifier.fit(X_train, Y_train)
    
    # from sklearn.model_selection import RepeatedStratifiedKFold
    # from sklearn.model_selection import cross_val_score
    # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # n_scores = cross_val_score(classifier, X_train, Y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    # for train, test in cv.split(X_train, Y_train):
    #     X = X_train[train]
    #     Y = Y_train[train]
    #     classifier.n_estimators = int(classifier.n_estimators * 1.25)
    #     forest_model = classifier.fit(X, Y)
    
    # from sklearn.model_selection import cross_validate
    # cv_results = cross_validate(classifier.fit(X_train, Y_train),
    #                              X_train, Y_train,
    #                              cv=5, scoring='balanced_accuracy')
        
    
    # %% report performance
    #print(f'Accuracy: {np.mean(n_scores)} {np.mean(n_scores)}')
    
    # Evaluate the performance of the model
    forest_score = forest_model.score(X_test,Y_test)
    forest_prediction = forest_model.predict(X_test)
    report = classification_report(Y_test, forest_prediction)
    print(report)
    
    # %% print final results
    print(F"Decision tree accuracy: {tree_score}\n")
    print(F"Random forests accuracy: {forest_score}")
        
        
    return (tree_model, forest_model)

# %% Function to find the optimal parameters

# Some possible scorers:
    # average_precision
    # f1
    # roc_auc

def hyperparameter_tuning(X_train, Y_train, X_test, Y_test,
                          score = 'balanced_accuracy', save_name = 'forest_model'):
    classifier =  RandomForestClassifier()
    
    #===Set the range of possible values for all parameters===#
        
    # Number of trees in random forest
    n_estimators = [50, 100, 200, 400, 800, 1000, 1400, 2000]
    
    # Number of features to consider at every split
    max_features = ['auto', 'None']
    
    # Maximum number of levels in tree
    max_depth = [10, 20, 30, 40, 60, 80, 110]
    
    # Grow trees with max_leaf_nodes in best-first fashion
    max_leaf_nodes = [10, 100]
    
    # Minimum number of samples required to split a node
    min_samples_split = [2, 10, 20]
    
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 4, 8]
    
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    
    # Wrap the parameter space in random_grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'max_leaf_nodes' : max_leaf_nodes,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    
    
    #===Run a randomised search for the optimal parameter setting===#
    classifier_random = RandomizedSearchCV(estimator = classifier,
                                           param_distributions = random_grid,
                                           n_iter = 100,
                                           cv = 5,
                                           verbose=2,
                                           random_state=42,
                                           n_jobs = -1,
                                           scoring = score)
    
    # Fit the random search model
    param_opt_rand = classifier_random.fit(X_train, Y_train)
    
    #=== Narrow down random optimal solutions to the best hyperparamets ===#
    # Extrapolate random optimal parameter values
    bootstrap = param_opt_rand.best_estimator_.bootstrap
    max_depth = param_opt_rand.best_estimator_.max_depth
    max_features = param_opt_rand.best_estimator_.max_features
    max_leaf_nodes = param_opt_rand.best_estimator_.max_leaf_nodes
    min_samples_leaf = param_opt_rand.best_estimator_.min_samples_leaf
    min_samples_split = param_opt_rand.best_estimator_.min_samples_split
    n_estimators = param_opt_rand.best_estimator_.n_estimators
    
    # IDEA: implement a precision variable. Divide the step size and the offset
    # used in the np.arange() by the precision which should by default be 1.
    # Note: if 'None' gets selected for some param there might be an error here
    param_grid = {
        'bootstrap': [bootstrap],
        'max_depth': [np.arange(max_depth-10, max_depth + 21, 10), None],
        'max_features': ['auto', 'sqrt'],
        'min_samples_leaf': np.arange(min_samples_leaf - 1 ,
                                      min_samples_leaf + 2),
        'max_leaf_nodes' : [np.arange(max_leaf_nodes - 10,
                                      max_leaf_nodes + 10, 2), None],
        'min_samples_split': np.arange(min_samples_split - 1,
                                       min_samples_split + 2),
        'n_estimators': np.arange(n_estimators - 200,
                                  n_estimators + 401,
                                  200)
    }
    
    # Find the optimal hyperparameters using gridsearch
    # initialise the grid search with cross validation
    classifier_gridsearch = GridSearchCV(estimator = classifier,
                                         param_grid = param_grid, 
                                         cv = 5, n_jobs = -1, verbose = 2,
                                         scoring = score)
    # Run the grid search to find the model with the optimal hyperparamterers
    classifier_gridsearch.fit(X_train, Y_train)
    # Extrapolate the optimal hyperparamters
    RandomForests_params_opt = classifier_gridsearch.best_params_
    
    
    #=== print and save the results ===#
    
    # Locally save results
    RandomForests_params_file = open("Hyperparameters/RandomForests.pkl", "wb")
    pickle.dump(RandomForests_params_opt, RandomForests_params_file)
    RandomForests_params_file.close()
    
    # Locally save the entire model
    joblib.dump(classifier_gridsearch.best_estimator_,
                f'Models/RandomForests/{save_name}.pkl')
    
    # Evaluate the optimal model
    final_accuracy = classifier_gridsearch.best_estimator_.score(X_test,
                                                                 Y_test)

    # print the results
    print("Random Forests hyperparamter optimisation completed succesfully.")
    print(f"Best random forests model accuracy: {final_accuracy}")
    
    return classifier_gridsearch


def select_features(X_train, Y_train, threshold = 'mean'):
    # Initialise the forst model with feature selection
    classifier =  SelectFromModel(RandomForestClassifier(),
                                  threshold = 'mean')
    # Setup the repeated Kfold cross validation
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)

    # Fit the model on the entire training data and initialise the matrices
    # of the cummulative feature importances and counts
    classifier.fit(X_train, Y_train)
    feat_importances = classifier.estimator_.feature_importances_
    # Feat_counts is currently redundant as it counts how often a feature is 
    # selected based on the 'SelectFromModel' threshold which does not support
    # a top n features. Feat_counts is therefore not used at the moment.
    feat_counts = classifier.get_support().astype(int)
    count = 0
    # Run through the cross validation loops abd record the feature selection
    for train, test in cv.split(X_train, Y_train):
        X = X_train[train]
        Y = Y_train.iloc[train]
        classifier.fit(X, Y)
        feat_importances += classifier.estimator_.feature_importances_
        feat_counts += classifier.get_support().astype(int)
        count += 1
    feat_importances = feat_importances/count
    
    # use_top is by default false such that a numeric threshold is used
    use_top = False
    
    # Select the important features
    if threshold == 'mean':
        threshold = np.mean(feat_importances)
    elif threshold == 'median':
        threshold = np.median(feat_importances)
    elif type(threshold) == str and threshold[0:3] == 'top':
        # If a top n features is specified, use_top will be set to True so
        top = int(threshold[3:len(threshold)])
        use_top = True
    else:
        try:
            threshold = float(threshold)
        except:
            raise Exception('Invalid theshold!')
    
    if use_top:
        # Set threshold to the lowest importance value in the top n features
        threshold = np.sort(feat_importances, axis = -1)[::-1][top-1]
    important_feat_index = feat_importances >= threshold
    
    return important_feat_index
    
 
