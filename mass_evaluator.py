# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 13:07:17 2020

@author: The Jipsess
"""

# %% Import packages
import os
import numpy as np
import pandas as pd
from sklearn import metrics
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
import time

# %% Test feature selection
thresholds = ['top10', 'top100', 'top200', 'top500', 'top1000', 'top10000', 'mean', 'median']

evaluation_mat = pd.DataFrame(columns = thresholds,
                              index = ['Features:', 'Accuracy:', 'AUROC:'])
starttime = time.time()
i = 0
for threshold in thresholds:
    X_train, Y_train, X_test, Y_test = init(file = 'all',
                                            label = 'DILI1',
                                            upsample = True,
                                            downsample = False)
    
    important_feat_index = Tree_prediction.select_features(X_train, Y_train,
                                                             threshold = threshold)
    
    [X_train, X_test] = feature_filter([X_train, X_test],
                                       important_feat_index)
    
    forest_model = Tree_prediction.hyperparameter_tuning(X_train, Y_train,
                                                         X_test, Y_test,
                                                         score = 'roc_auc',
                                                         save_name = 'forest_model_all_10')
    
    accuracy = metrics.accuracy_score(Y_test,
                                  forest_model.best_estimator_.predict(X_test))
    AUROC = metrics.roc_auc_score(Y_test,
                                  forest_model.best_estimator_.predict(X_test))
    
    evaluation_mat.iloc[0, i] = threshold
    evaluation_mat.iloc[1, i] = accuracy
    evaluation_mat.iloc[2, i] = AUROC
    i += 1
    print(f'finnished iteration for threshold setting: {threshold}')
    print(f'accuracy: {accuracy}, AUROC: {AUROC}')
    print(f'Time elapsed: {(time.time() - starttime)/60:.0f} minutes')

# %% test different data
files = ['p3-phh-camda2020.csv',
        'p4-hepg2-camda2020.csv',
        'p5-haie-camda2020.csv',
        'p6-a375-camda2020.csv',
        'p7-mcf7-camda2020.csv',
        'p8-pc3-camda2020.csv']

columns = []
done = []
for file in files:
    for file2 in files:
        if file == file2:
            continue
        if file2 in done:
            continue
        columns.append(file[0:2]+' & ' + file2[0:2])
        done.append(file)


evaluation_mat = pd.DataFrame(columns = columns,
                              index = ['Features:', 'Accuracy:', 'AUROC:'])

starttime = time.time()
i = 0
for file in files:
    for file2 in files:
        X_train, Y_train, X_test, Y_test = init(file = [file, file2],
                                                label = 'DILI1',
                                                upsample = True,
                                                downsample = False)
        
        important_feat_index = Tree_prediction.select_features(X_train, Y_train,
                                                                 threshold = 'top1000')
        
        [X_train, X_test] = feature_filter([X_train, X_test],
                                           important_feat_index)
        
        forest_model = Tree_prediction.hyperparameter_tuning(X_train, Y_train,
                                                             X_test, Y_test,
                                                             score = 'roc_auc',
                                                             save_name = 'forest_model_all_10')
        
        accuracy = metrics.accuracy_score(Y_test,
                                      forest_model.best_estimator_.predict(X_test))
        AUROC = metrics.roc_auc_score(Y_test,
                                      forest_model.best_estimator_.predict(X_test))
        
        evaluation_mat.iloc[0, i] = file[0:2]+' & ' + file2[0:2]
        evaluation_mat.iloc[1, i] = accuracy
        evaluation_mat.iloc[2, i] = AUROC
        i += 1
        print(f'finnished iteration for data file: {file[0:2] + file2[0:2]}')
        print(f'accuracy: {accuracy}, AUROC: {AUROC}')
        print(f'Iteration {i}/15 and time elapsed: {(time.time() - starttime)/60:.0f} minutes\n\n')