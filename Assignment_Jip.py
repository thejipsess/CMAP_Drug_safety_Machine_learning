# -*- coding: utf-8 -*-
"""
Created on Thu May 14 11:50:56 2020

@author: The Jipsess
"""


# %% Import packages
import os
# %% Set the working directory
# IMPORTNAT!! make sure to set the working directory to the path where you
# stored the python files, including this script and the Data & Models folders.
os.chdir('D:/OneDrive/school/1. Master/8. Machine Learning and Multivariate '
         + 'Statistics/Assignment/CMAP_Drug_safety_Machine_learning')

# Load local scripts
from Data import init, feature_filter
import Tree_prediction
import SVM_prediction
import ANN

# %% Initialise the data
X_train, Y_train, X_test, Y_test = init(file = 'p3-phh-camda2020.csv',
                                        label = 'DILI1',
                                        upsample = True,
                                        downsample = False)

# %% Feature Selection
important_feat_index = Tree_prediction.select_features(X_train, Y_train,
                                                         threshold = 'top1000')

# Filter all the data on the feature selection
[X_train, X_test] = feature_filter([X_train, X_test],
                                   important_feat_index)

# %% Random Forest
forest_model = Tree_prediction.hyperparameter_tuning(X_train, Y_train,
                                                     X_test, Y_test,
                                                     score = 'roc_auc',
                                                     save_name = 'forest_model')

# %% Support Vector machine
SVM_model =  SVM_prediction.hyperparameter_tuning(X_train, Y_train,
                                                  X_test, Y_test,
                                                      save_name = 'SVM_model')

# %% Artificial Neural Network
ANN_model = ANN.hyperparameter_tuning(X_train, Y_train, X_test, Y_test,
                                      balance_class_weights = False)