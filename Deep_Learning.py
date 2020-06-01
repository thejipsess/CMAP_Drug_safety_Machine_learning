# -*- coding: utf-8 -*-
"""
Created on Tue May  5 15:07:56 2020

@author: The Jipsess
"""

# %% Set the working directory
import os
# IMPORTNAT!! make sure to set the working directory to the path where you
# stored the python files, including this script.
os.chdir('C:/Users/jipde/OneDrive/school/1. Master/8. Machine Learning and Multivariate '
         + 'Statistics/Assignment')


# %% Load all the required packages
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# %% Loading & pre-processing of the data
X = pd.read_csv('Data/p7-mcf7-camda2020.csv')
Y = pd.read_csv('Data/targets-camda2020.csv')

# Only keep samples for which training data is available.
Y = Y.loc[Y.CAM_ID.isin(X.CAM_ID)]

# Create identical row order for X & Y
X = X.sort_values('CAM_ID').reset_index(drop = True)
Y = Y.sort_values('CAM_ID').reset_index(drop = True)

# Ensure that X and Y have identical CAM_IDs in each row
if not Y.CAM_ID.equals(X.CAM_ID):
    raise Exception('X != Y...  make sure the rows of X and Y describe the ' +
                    'same samples!!')

# Seperate the validation sets
Y_val = Y.loc[Y.Training_Validation != 'Training Set']
X_val = X.loc[X.CAM_ID.isin(Y_val.CAM_ID)]

# Remove validation sets from the test and training sets
Y = Y.loc[Y.Training_Validation == 'Training Set']
X = X.loc[X.CAM_ID.isin(Y.CAM_ID)]

# Turn ID column into rownames
X.set_index('CAM_ID', inplace = True, verify_integrity = True)
Y.set_index('CAM_ID', inplace = True, verify_integrity = True)

# Create the test and training set
X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                test_size = 0.2,
                                                random_state =
                                                np.random.randint(0, 10000))

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# %% Up-sample the minority class in the data to balance the classes
from sklearn.utils import resample

df = pd.concat([X, Y], axis = 1)

# Separate majority and minority classes
df_majority = df[df.DILI1==0]
df_minority = df[df.DILI1==1]
 
# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,                      # sample with replacement
                                 n_samples=df_majority.shape[0],    # to match majority class
                                 random_state=1)                  # reproducible results
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
 
# Display new class counts
df_upsampled.DILI1.value_counts()

X = df_upsampled.iloc[:,0:12329]
Y = df_upsampled.iloc[:,12329:12337]
X.drop('CAM_ID',axis = 1, inplace = True)
Y.drop('CAM_ID',axis = 1, inplace = True)

# %% Build the Artificial Neural Network.

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
classifier.add(Dropout(rate = 0.1))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train.DILI1, batch_size = 100, epochs = 100)

# %% evaluate model.

# Predicting the Test set results
Y_pred = classifier.predict(X_test)
Y_pred[Y_pred > 0.5] = 1
Y_pred[Y_pred != 1] = 0

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test.DILI1, Y_pred)

def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements 

accuracy(cm)

# %% Model Parameter optimisation

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2


def build_classifier(tuned_optimiser):
    
    classifier = Sequential()
    classifier.add(Dense(units = 64,
                         kernel_initializer = 'uniform',
                         activation = 'relu',
                         kernel_regularizer = l2(0.1),
                         input_dim = len(X.columns)))
    
    classifier.add(Dense(units = 64,
                         kernel_initializer = 'uniform',
                         kernel_regularizer = l2(0.1),
                         activation = 'relu'))
    
    classifier.add(Dense(units = 64,
                         kernel_initializer = 'uniform',
                         kernel_regularizer = l2(0.1),
                         activation = 'relu'))
    
    classifier.add(Dense(units = 1,
                         kernel_initializer = 'uniform',
                         activation = 'sigmoid'))
    
    classifier.compile(optimizer = tuned_optimiser,
                       loss = 'binary_crossentropy',
                       metrics = ['binary_accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier,
                             batch_size = 10,
                             epochs = 100)

# Set class weight to account for imbalanced data
one_count = np.sum(y_train.DILI1)
zero_count = y_train.DILI1.shape[0] - one_count
one_weight = one_count / y_train.DILI1.shape[0]
zero_weight = zero_count / y_train.DILI1.shape[0]
classweight = {0 : zero_weight,
               1 : one_weight}

parameters = {'batch_size': [100, 240],
              'epochs': [10, 100],
              'tuned_optimiser': ['adam',  'SGD']}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'roc_auc',
                           cv = 5,
                           n_jobs = 1)

grid_search = grid_search.fit(X_train,y_train.DILI1, class_weight = classweight)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
best_model = grid_search.best_estimator_.model

# 

# Predicting the Test set results
y_pred = grid_search.best_estimator_.model.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test.DILI1, y_pred)

def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements 

accuracy(cm)